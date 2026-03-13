/* Write
     -Laplace(p) = f
   as
      J       = grad(p)
      -div(J) = f
   Weak formulation
     (J, Jt) + (div(Jt),p) = 0
               (div(J),pt) = -(f,pt)
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

//#include <deal.II/lac/petsc_communication_pattern.h>
//#include <deal.II/lac/exceptions.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <petscpctypes.h>
#include <petscpc.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#define AssertPETSc(code)                          \
   do                                              \
   {                                               \
      PetscErrorCode ierr = (code);                \
      AssertThrow(ierr == 0, ExcPETScError(ierr)); \
   } while (false)

namespace LA
{
   using namespace dealii::PETScWrappers;
}

#define DIRICHLET 1
#define NEUMANN   2

#include <fstream>
#include <iostream>

using namespace dealii;

#if PROBLEM == DIRICHLET
#include "dirichlet.h"
const std::string problem = "Dirichlet";
#elif PROBLEM == NEUMANN
#include "neumann.h"
const std::string problem = "Neumann";
#else
#error "Unknown PROBLEM"
#endif

struct BlockVariables
{
   LA::MPI::BlockSparseMatrix system_matrix;
   LA::MPI::BlockVector       system_rhs;
   LA::MPI::BlockVector       solution;
};

struct Variables
{
   LA::MPI::SparseMatrix system_matrix;
   LA::MPI::Vector       system_rhs;
   LA::MPI::Vector       solution;
};

//------------------------------------------------------------------------------
template <int dim>
class Postprocessor : public DataPostprocessor<dim>
{
public:
   Postprocessor();

   void
   evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& input_data,
                         std::vector<Vector<double>>& computed_quantities) const override
   {
      const std::vector<std::vector<Tensor<1, dim>>>& uh = input_data.solution_gradients;
      for(unsigned int i = 0; i < uh.size(); ++i)
      {
         computed_quantities[i](0) = uh[i][0][0] + uh[i][1][1];
         if(dim == 3) computed_quantities[i](0) += uh[i][2][2];
      }
   }

   std::vector<std::string>
   get_names() const override
   {
      return {"div_j"};
   }

   UpdateFlags
   get_needed_update_flags() const override
   {
      return update_gradients;
   }
};

//------------------------------------------------------------------------------
template <int dim>
Postprocessor<dim>::Postprocessor()
   : DataPostprocessor<dim>()
{}

//------------------------------------------------------------------------------
template<typename PreconditionerType>
class MassMatrixInverse
{
public:
  MassMatrixInverse(const LA::MPI::SparseMatrix &M_in, SolverCG<LA::MPI::Vector> &solver, PreconditionerType& preconditioner)
    : M(M_in), solver(solver), preconditioner(preconditioner)
  {}

  // mutable int solve_iter;
  
  void vmult(LA::MPI::Vector &dst,
             const LA::MPI::Vector &src) const
  {
    // ReductionControl reduction_control(2000, 1e-18, 1e-10);
    // SolverCG<LA::MPI::Vector> solver(reduction_control);
    //
    // LA::PreconditionJacobi preconditioner;
    // preconditioner.initialize(M);

    solver.solve(M, dst, src, preconditioner);
    // solve_iter = reduction_control.last_step();
  }

private:
  const LA::MPI::SparseMatrix &M;
  SolverCG<LA::MPI::Vector> &solver;
  PreconditionerType& preconditioner;
};

//------------------------------------------------------------------------------
template<typename MatrixType>
class SchurMatrix
{
public:
  SchurMatrix(const MatrixType & op_M_inv, const LA::MPI::SparseMatrix &B, const LA::MPI::Vector &U)
    : op_M_inv(op_M_inv),
    B(B), 
    U(U)
  {}

  void vmult(LA::MPI::Vector &dst,
             const LA::MPI::Vector &src) const
  {
      LA::MPI::Vector tmp_1;
      tmp_1.reinit(U);
      LA::MPI::Vector tmp_2;
      tmp_2.reinit(U);
      B.vmult(tmp_1, src);
      op_M_inv.vmult(tmp_2, tmp_1);
      B.Tvmult(dst, tmp_2);
  }

private:
  const MatrixType & op_M_inv;
  const LA::MPI::SparseMatrix &B;
  const LA::MPI::Vector &U;
};
//------------------------------------------------------------------------------
template<typename MatrixType, typename PreconditionerType, typename Solver_Control>
class SchurMatrixInverse
{
public:
  SchurMatrixInverse(SchurMatrix<MatrixType> & op_S,
                     PreconditionerType & preconditioner,
                     const Solver_Control & solver_control_S)
    : op_S(op_S),
      preconditioner(preconditioner),
      solver_control_S(solver_control_S),
      solver(this->solver_control_S)
  {}

  mutable int solve_iter;

  void vmult(LA::MPI::Vector &dst,
             const LA::MPI::Vector &src) const
  {
      solver.solve(op_S, dst, src, preconditioner);
      solve_iter = solver_control_S.last_step();
  }

private:
  const SchurMatrix<MatrixType> & op_S;
  PreconditionerType &preconditioner;

  mutable Solver_Control solver_control_S;
  mutable SolverCG<LA::MPI::Vector> solver;
};
// template<typename MatrixType, typename PreconditionerType>
// class SchurMatrixInverse
// {
// public:
//   SchurMatrixInverse(SchurMatrix<MatrixType> & op_S, PreconditionerType & preconditioner,  int& iterations)
//     : op_S(op_S), preconditioner(preconditioner), iterations(iterations)
//   {}
//
//   mutable int solve_iter;
//
//   void vmult(LA::MPI::Vector &dst,
//              const LA::MPI::Vector &src) const
//   {
//
//    //  SolverControl            solver_control_S(src.size(), 1.e-8);
//    // SolverCG<LA::MPI::Vector> solver_S(solver_control_S);
//    // solver_S.solve(op_S, dst, src, preconditioner);
//    // solve_iter = solver_control_S.last_step();
//    if (iterations != 30) 
//    {
//         SolverControl            solver_control_S(iterations, 1.e-8);
//        SolverCG<LA::MPI::Vector> solver_S(solver_control_S);
//        solver_S.solve(op_S, dst, src, preconditioner);
//        solve_iter = solver_control_S.last_step();
//    }
//    else 
//    {
//        // std::cout << iterations << std::endl;
//        IterationNumberControl  solver_control_S(iterations, 1.e-18);
//        // std::cout << iterations << std::endl;
//        // IterationNumberControl  solver_control_S(src.size(), 1.e-8);
//        SolverCG<LA::MPI::Vector> solver_S(solver_control_S);
//        solver_S.solve(op_S, dst, src, preconditioner);
//        solve_iter = solver_control_S.last_step();
//    }
//   }
//
// private:
//   const SchurMatrix<MatrixType> & op_S;
//   PreconditionerType &preconditioner;
//   int &iterations;
// };
//------------------------------------------------------------------------------
// template<LA::MPI::BlockSparseMatrix, LA::MPI::BlockVector>
template<typename MatrixType, typename PreconditionerType>
// template<typename PreconditionerType>
class BlockSchurPreconditioner
{
public:
  BlockSchurPreconditioner(LA::MPI::SparseMatrix& B, PreconditionerType &preconditioner_M, MatrixType &op_aS_inv)
      : B(B), preconditioner_M(preconditioner_M), op_aS_inv(op_aS_inv)
  {}

  void vmult(LA::MPI::BlockVector &dst,
             const LA::MPI::BlockVector &src) const
  {
   const auto &b_1 = src.block(0);
   const auto &b_2 = src.block(1);

   auto &y_1 = dst.block(0);
   auto &y_2 = dst.block(1);

  preconditioner_M.vmult(y_1, b_1);
  
  LA::MPI::Vector tmp_1;
  tmp_1.reinit(b_2);
  B.Tvmult(tmp_1, y_1);
  tmp_1.add(-1.0, b_2);
  // tmp_1.sadd(-1.0, 1.0, P);
  // tmp_1 -= P;
  op_aS_inv.vmult(y_2, tmp_1);
  }

private:
  LA::MPI::SparseMatrix &B;
  PreconditionerType &preconditioner_M;
  MatrixType & op_aS_inv;
};
//------------------------------------------------------------------------------
class ParameterReader : public EnableObserverPointer
{
public:
  ParameterReader(ParameterHandler&);
  void read_parameters(const std::string&);

private:
  void declare_parameters();
  ParameterHandler& prm;
};

//------------------------------------------------------------------------------
ParameterReader::ParameterReader(ParameterHandler& paramhandler)
  : prm(paramhandler)
{}

void
ParameterReader::declare_parameters()
{
  prm.declare_entry("degree",
                    "1",
                    Patterns::Integer(0),
                    "Degree of polynomial");
  prm.declare_entry("nrefine",
                    "4",
                    Patterns::Integer(0),
                    "number of refinements");
  prm.declare_entry("initial_refine",
                    "4",
                    Patterns::Integer(0),
                    "Initial refinement of mesh");
  prm.declare_entry("perturb grid",
                    "0",
                    Patterns::Double(0,1),
                    "Perturb grid");
  prm.declare_entry("solver",
                    "schur",
                    Patterns::Selection("schur|umfpack|petsc_gmres|gmres"),
                    "Linear solver");
}

void
ParameterReader::read_parameters(const std::string& parameter_file)
{
  declare_parameters();
  prm.parse_input(parameter_file);
}

//------------------------------------------------------------------------------
template <int dim>
class MixedLaplaceProblem
{
public:
   MixedLaplaceProblem(const unsigned int nrefine,
                       const unsigned int degree,
                       const unsigned int initial_refine,
                       const double       perturb,
                       const std::string  solver);
   void run(std::vector<int>&    ncell,
            std::vector<double>& h_array,
            std::vector<int>&    phi_dofs,
            std::vector<int>&    j_dofs,
            std::vector<double>& phi_error,
            std::vector<double>& j_error,
            std::vector<double>& d_error,
            std::vector<int>&    phi_iterations,
            std::vector<int>&    j_iterations,
            std::vector<double>& phi_time,
            std::vector<double>& j_time);

private:
   using PTriangulation = parallel::distributed::Triangulation<dim>;

   void make_grid(const unsigned int refine);
   void setup_blockvar_system();
   void setup_var_system();
   void setup_system();
   template<typename VariableStruct> void assemble_system(VariableStruct &VarStruct);
   void solve_schur(int&    phi_iteration,
                    int&    j_iteration,
                    double& phi_time,
                    double& j_time);
   void solve_umfpack(int&    phi_iteration,
                      int&    j_iteration,
                      double& phi_time,
                      double& j_time);
   void solve_petsc_gmres(int&    phi_iteration,
                    int&    j_iteration,
                    double& phi_time,
                    double& j_time);
   void solve_gmres(int&    phi_iteration,
                    int&    j_iteration,
                    double& phi_time,
                    double& j_time);
   void solve(int&    phi_iteration,
              int&    j_iteration,
              double& phi_time,
              double& j_time);
   template<typename VariableStruct>void compute_errors(double& phi_err,
                       double& j_err,
                       double& d_err,
                       VariableStruct &VarStruct);
   template<typename VariableStruct>void output_results(VariableStruct &VarStruct);
   void refine_grid(unsigned int refine);

   MPI_Comm               mpi_comm;
   const unsigned int     mpi_rank;
   ConditionalOStream     pcout;

   const unsigned int degree;
   const unsigned int nrefine;
   const unsigned int initial_refine;
   const double       perturb;
   const std::string  linear_solver;

   Timer               timer;
   double              h_max;
   PTriangulation      triangulation;
   const FESystem<dim> fe;
   DoFHandler<dim>     dof_handler;

   AffineConstraints<double>  constraints;
   LA::MPI::BlockSparseMatrix system_matrix;
   LA::MPI::BlockVector       solution;
   LA::MPI::BlockVector       system_rhs;

   BlockVariables      block_vars;
   Variables           vars;
};

//------------------------------------------------------------------------------
template <int dim>
MixedLaplaceProblem<dim>::MixedLaplaceProblem(
      const unsigned int nrefine,
      const unsigned int degree,
      const unsigned int initial_refine,
      const double       perturb,
      const std::string  solver)
 : mpi_comm(MPI_COMM_WORLD),
   mpi_rank(Utilities::MPI::this_mpi_process(mpi_comm)),
   pcout(std::cout, mpi_rank==0),
   degree(degree),
   nrefine(nrefine),
   initial_refine(initial_refine),
   perturb(perturb),
   linear_solver(solver),
   triangulation(mpi_comm),
   fe(FE_RaviartThomas<dim>(degree), FE_DGQ<dim>(degree)),
   dof_handler(triangulation)
{}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::make_grid(const unsigned int refine)
{
   triangulation.clear();
   GridGenerator::hyper_cube(triangulation, 0.0, 1.0, false);
   triangulation.refine_global(refine);

   if(perturb <= 0.0) return;
   pcout << "Randomly perturbing grid with amplitude "
             << perturb << std::endl;

   // Randomly perturb the grid
   // TODO: Does this work in parallel
   const auto h = 1.0 / pow(2, refine);
   auto v = triangulation.begin_vertex();
   for(; v < triangulation.end_vertex(); ++v)
   {
      auto& p = v->vertex();
      if(p[0] > 0 && p[0] < 1 && p[1] > 0 && p[1] < 1)
      {
         double a = ((double)rand()) / RAND_MAX;
         double b = ((double)rand()) / RAND_MAX;
         p[0] +=  perturb * (2 * a - 1) * h;
         p[1] +=  perturb * (2 * b - 1) * h;
      }
   }
}

//------------------------------------------------------------------------------
template<int dim>
void
MixedLaplaceProblem<dim>::setup_blockvar_system()
{
    timer.start();
   dof_handler.distribute_dofs(fe);

   DoFRenumbering::component_wise(dof_handler);

   const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
   const unsigned int n_c = dofs_per_component[0],
                      n_p = dofs_per_component[dim];

   pcout << "Number of active cells: " << triangulation.n_active_cells()
             << std::endl
             << "Total number of cells: " << triangulation.n_cells()
             << std::endl
             << "Number of degrees of freedom: " << dof_handler.n_dofs()
             << " (" << n_c << '+' << n_p << ')' << std::endl;

   const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
   IndexSet locally_relevant_dofs =
       DoFTools::extract_locally_relevant_dofs(dof_handler);

   constraints.clear();
   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

#if PROBLEM == DIRICHLET
   // p = 0 on all boundary
   // Nothing to do, weakly imposed, just dont add any boundary integral.
#else // Neumann problem: TODO
   // J.n = 0 on all boundaries
   pcout << "hello 1" << std::endl;
   VectorTools::project_boundary_values_div_conforming
        (dof_handler,
         0,
         Functions::ZeroFunction<dim>(dim),
         types::boundary_id(0),
         constraints);

   // Add mean value constraint on phi
   Vector<double> integral_vector;
   integral_vector.reinit(dof_handler.n_dofs());

   const FEValuesExtractors::Scalar scalar_extractor(dim);
   const ComponentMask scalar_mask = fe.component_mask(scalar_extractor);
   std::vector<bool> bool_boundary_dofs;
   pcout << "hello 2" << std::endl;
   DoFTools::extract_dofs_with_support_on_boundary(dof_handler,
                                                   scalar_mask,
                                                   bool_boundary_dofs,
                                                   {0});
   pcout << "hello 2" << std::endl;

   const IndexSet all_dofs = DoFTools::extract_dofs(dof_handler, scalar_mask);
   types::global_dof_index first_dof = all_dofs.nth_index_in_set(0);

   const QGauss < dim - 1 > quadrature_formula(degree + 2);
   FEFaceValues<dim> face_fe_values(fe, quadrature_formula,
                                    update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);

   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
   const unsigned int face_n_q_points = quadrature_formula.size();
   Vector<double> local_rhs(dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

   pcout << "hello 3" << std::endl;
   for(const auto& cell : dof_handler.active_cell_iterators())
       if(cell -> is_locally_owned())
   {
      local_rhs = 0.0;
      for(unsigned int f = 0; f < cell->n_faces(); ++f)
      {
         if(cell->face(f)->at_boundary())
         {
            face_fe_values.reinit(cell, f);
            for(unsigned int q_point = 0; q_point < face_n_q_points; ++q_point)
            {
               for(unsigned int i = 0; i < dofs_per_cell; ++i)
               {
                  local_rhs(i) += face_fe_values[scalar_extractor].value(i, q_point) *
                                  face_fe_values.JxW(q_point);
               }
            }
         }
      }
      cell->get_dof_indices(local_dof_indices);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
         if(local_rhs(i) != 0)
         {
            integral_vector(local_dof_indices[i]) += local_rhs(i);
         }
      }
   }
   pcout << "hello 4" << std::endl;

   std::vector<std::pair<types::global_dof_index, double>> rhs;
   for(const types::global_dof_index i : all_dofs)
      if(i != first_dof && bool_boundary_dofs[i])
         rhs.emplace_back(i, -integral_vector(i) / integral_vector(first_dof));
   pcout << "hello 5" << std::endl;
   constraints.add_constraint(first_dof, rhs);
#endif

   constraints.close();

   Table<2, DoFTools::Coupling> coupling;
   coupling.reinit(dim + 1, dim + 1);
   coupling.fill(DoFTools::always);
   coupling(dim, dim) = DoFTools::none;
   std::vector<IndexSet>
       owned_partitioning = {locally_owned_dofs.get_view(0, n_c),
                             locally_owned_dofs.get_view(n_c, n_c + n_p)};
   std::vector<IndexSet>
      relevant_partitioning = {locally_relevant_dofs.get_view(0, n_c),
                               locally_relevant_dofs.get_view(n_c, n_c + n_p)};

   BlockDynamicSparsityPattern sparsity_pattern(relevant_partitioning);

   DoFTools::make_sparsity_pattern(dof_handler,
                                   coupling,
                                   sparsity_pattern,
                                   constraints,
                                   false);
   SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                              locally_owned_dofs,
                                              mpi_comm,
                                              locally_relevant_dofs);
   block_vars.system_matrix.reinit(owned_partitioning,
                        sparsity_pattern,
                        mpi_comm);
   block_vars.solution.reinit(owned_partitioning, relevant_partitioning, mpi_comm);
   block_vars.system_rhs.reinit(owned_partitioning, mpi_comm);;
   timer.stop();
}
//------------------------------------------------------------------------------
template<int dim>
void
MixedLaplaceProblem<dim>::setup_var_system()
{
    timer.start();
   dof_handler.distribute_dofs(fe);

   // DoFRenumbering::component_wise(dof_handler);

   // const std::vector<types::global_dof_index> dofs_per_component =
   //    DoFTools::count_dofs_per_fe_component(dof_handler);
   // const unsigned int n_c = dofs_per_component[0],
   //                    n_p = dofs_per_component[dim];

   pcout << "Number of active cells: " << triangulation.n_active_cells()
             << std::endl
             << "Total number of cells: " << triangulation.n_cells()
             << std::endl
             << "Number of degrees of freedom: " << dof_handler.n_dofs() <<std::endl;
             // << " (" << n_c << '+' << n_p << ')' << std::endl;

   const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
   IndexSet locally_relevant_dofs =
       DoFTools::extract_locally_relevant_dofs(dof_handler);

   constraints.clear();
   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

#if PROBLEM == DIRICHLET
   // p = 0 on all boundary
   // Nothing to do, weakly imposed, just dont add any boundary integral.
#else // Neumann problem: TODO
   // J.n = 0 on all boundaries
   VectorTools::project_boundary_values_div_conforming
        (dof_handler,
         0,
         Functions::ZeroFunction<dim>(dim),
         types::boundary_id(0),
         constraints);

   // Add mean value constraint on phi
   Vector<double> integral_vector;
   integral_vector.reinit(dof_handler.n_dofs());

   const FEValuesExtractors::Scalar scalar_extractor(dim);
   const ComponentMask scalar_mask = fe.component_mask(scalar_extractor);
   std::vector<bool> bool_boundary_dofs;
   DoFTools::extract_dofs_with_support_on_boundary(dof_handler,
                                                   scalar_mask,
                                                   bool_boundary_dofs,
                                                   {0});

   const IndexSet all_dofs = DoFTools::extract_dofs(dof_handler, scalar_mask);
   types::global_dof_index first_dof = all_dofs.nth_index_in_set(0);

   const QGauss < dim - 1 > quadrature_formula(degree + 2);
   FEFaceValues<dim> face_fe_values(fe, quadrature_formula,
                                    update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);

   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
   const unsigned int face_n_q_points = quadrature_formula.size();
   Vector<double> local_rhs(dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

   for(const auto& cell : dof_handler.active_cell_iterators())
   {
      local_rhs = 0.0;
      for(unsigned int f = 0; f < cell->n_faces(); ++f)
      {
         if(cell->face(f)->at_boundary())
         {
            face_fe_values.reinit(cell, f);
            for(unsigned int q_point = 0; q_point < face_n_q_points; ++q_point)
            {
               for(unsigned int i = 0; i < dofs_per_cell; ++i)
               {
                  local_rhs(i) += face_fe_values[scalar_extractor].value(i, q_point) *
                                  face_fe_values.JxW(q_point);
               }
            }
         }
      }
      cell->get_dof_indices(local_dof_indices);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
         if(local_rhs(i) != 0)
         {
            integral_vector(local_dof_indices[i]) += local_rhs(i);
         }
      }
   }

   std::vector<std::pair<types::global_dof_index, double>> rhs;
   for(const types::global_dof_index i : all_dofs)
      if(i != first_dof && bool_boundary_dofs[i])
         rhs.emplace_back(i, -integral_vector(i) / integral_vector(first_dof));
   constraints.add_constraint(first_dof, rhs);
#endif

   constraints.close();

   Table<2, DoFTools::Coupling> coupling;
   coupling.reinit(dim + 1, dim + 1);
   coupling.fill(DoFTools::always);
   coupling(dim, dim) = DoFTools::none;
   DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
   DoFTools::make_sparsity_pattern(dof_handler,
                                   coupling,
                                   sparsity_pattern,
                                   constraints,
                                   false);
   SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                              locally_owned_dofs,
                                              mpi_comm,
                                              locally_relevant_dofs);
   vars.system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern,
                        mpi_comm);
   // vars.solution.reinit(locally_owned_dofs, mpi_comm);
   vars.system_rhs.reinit(locally_owned_dofs, mpi_comm);;
   vars.solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
   // vars.system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);;

   timer.stop();
}
//------------------------------------------------------------------------------
template<int dim>
void
MixedLaplaceProblem<dim>::setup_system()
{
   timer.start();

   if(linear_solver == "schur" || linear_solver == "gmres")
   {
       setup_blockvar_system();
   }else{
       setup_var_system();
    }
   // // sparsity_pattern_1.copy_from(sparsity_pattern);
   //
   // for(unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
   // {
   //     for(unsigned int j = 0; j < dof_handler.n_dofs(); ++j)
   //     {
   //         if(sparsity_pattern.exists(i,j))
   //         sparsity_pattern_1.add(i, j);
   //     }
   // }

   timer.stop();
   // double time_elapsed = timer.last_wall_time();
}

//------------------------------------------------------------------------------
template <int dim>
template<typename VariableStruct>
void
MixedLaplaceProblem<dim>::assemble_system(VariableStruct &VarStruct)
{
   timer.start();
   const QGauss<dim> quadrature_formula(degree + 2);
   FEValues<dim>  fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

   const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
   const unsigned int n_q_points      = quadrature_formula.size();

   FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
   Vector<double>     local_rhs(dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   const PrescribedSolution::RHSFunction<dim> rhs_function;
   const FEValuesExtractors::Vector current(0);
   const FEValuesExtractors::Scalar potential(dim);

   h_max = 0.0;
   double aspect = 1.0;
   for(const auto& cell : dof_handler.active_cell_iterators())
   if(cell->is_locally_owned())
   {
      h_max = std::max(h_max, cell->diameter());
      const double dx = cell->face(2)->measure();
      const double dy = cell->face(0)->measure();
      const double emin = std::min(dx, dy);
      const double emax = std::max(dx, dy);
      aspect = std::max(aspect, emax / emin);

      fe_values.reinit(cell);

      local_matrix = 0.0;
      local_rhs    = 0.0;

      for(unsigned int q = 0; q < n_q_points; ++q)
      {
         const auto rhs = rhs_function.value(fe_values.quadrature_point(q));
         for(unsigned int i = 0; i < dofs_per_cell; ++i)
         {
            const Tensor<1, dim> phi_i_c = fe_values[current].value(i, q);
            const double div_phi_i_c = fe_values[current].divergence(i, q);
            const double phi_i_p     = fe_values[potential].value(i, q);

            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
               const Tensor<1, dim> phi_j_c =
                  fe_values[current].value(j, q);
               const double div_phi_j_c =
                  fe_values[current].divergence(j, q);
               const double phi_j_p = fe_values[potential].value(j, q);

               local_matrix(i, j) +=
                  (phi_i_c * phi_j_c
                   + phi_i_p * div_phi_j_c
                   + div_phi_i_c * phi_j_p)
                  * fe_values.JxW(q);
            }

            local_rhs(i) += - phi_i_p * rhs * fe_values.JxW(q);
         }
      }

      cell->get_dof_indices(local_dof_indices);
      // if(linear_solver == "schur" || linear_solver == "gmres")
      // {
      // constraints.distribute_local_to_global(local_matrix,
      //                                        local_rhs,
      //                                        local_dof_indices,
      //                                        block_vars.system_matrix,
      //                                        block_vars.system_rhs);
      // }
      // else
      // {
      constraints.distribute_local_to_global(local_matrix,
                                             local_rhs,
                                             local_dof_indices,
                                             VarStruct.system_matrix,
                                             VarStruct.system_rhs);
      // }
   }
   // if(linear_solver == "schur" || linear_solver == "gmres")
   // {
   //     block_vars.system_matrix.compress(VectorOperation::add);
   //     block_vars.system_rhs.compress(VectorOperation::add);
   // }
   // else{
   VarStruct.system_matrix.compress(VectorOperation::add);
   VarStruct.system_rhs.compress(VectorOperation::add);
   // }
   timer.stop();
   double time_elapsed = timer.last_wall_time();

   // TODO reduce h_max, aspect over all ranks
   pcout << "Largest cell aspect ratio = " << aspect << std::endl;
   pcout << "Time for assembly = " << time_elapsed << " sec\n";
}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::solve_schur(int&    phi_iteration,
                                      int&    j_iteration,
                                      double& phi_time,
                                      double& j_time)
{
   
   const auto& M = block_vars.system_matrix.block(0, 0);
   const auto& B = block_vars.system_matrix.block(0, 1);

   const auto& F = block_vars.system_rhs.block(0);
   const auto& G = block_vars.system_rhs.block(1);

   const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
   const unsigned int n_c = dofs_per_component[0],
                      n_p = dofs_per_component[dim];
   const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
   std::vector<IndexSet>
       owned_partitioning = {locally_owned_dofs.get_view(0, n_c),
                             locally_owned_dofs.get_view(n_c, n_c + n_p)};
   LA::MPI::BlockVector distributed_solution(owned_partitioning, mpi_comm);
   // auto& U = block_vars.solution.block(0);
   // auto& P = block_vars.solution.block(1);
   auto & U = distributed_solution.block(0);
   auto & P = distributed_solution.block(1);

   // TrilinosWrappers::SparseMatrix M_1;

   // const auto op_M = TrilinosWrappers::linear_operator(M_1); \\ why is this not working??
   // const auto op_M = linear_operator<LA::MPI::Vector>(M);
   // const auto op_B = linear_operator<LA::MPI::Vector>(B);

   // ReductionControl         reduction_control_M(2000, 1.0e-18, 1.0e-10);
   // SolverCG<LA::MPI::Vector> solver_M(reduction_control_M);
   LA::PreconditionJacobi preconditioner_M;

   preconditioner_M.initialize(M);
    ReductionControl reduction_control(2000, 1e-18, 1e-10);
    SolverCG<LA::MPI::Vector> solver(reduction_control);

   MassMatrixInverse<LA::PreconditionJacobi> op_M_inv(M,solver, preconditioner_M);
   // const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

   // const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
   SchurMatrix<MassMatrixInverse<LA::PreconditionJacobi>> op_S(op_M_inv, B, U);
   SchurMatrix<LA::PreconditionJacobi> op_aS(preconditioner_M, B, U);
  PreconditionIdentity identity;

  // SchurMatrixInverse< SchurMatrix<LA::PreconditionJacobi>, PreconditionIdentity>op_aS_inv(op_aS, identity, 30);
  // int iterations = 30;
  IterationNumberControl iterations(30, 1e-18);
  SchurMatrixInverse op_aS_inv(op_aS, identity, iterations);
   // const auto op_aS =
   //    transpose_operator(op_B) * linear_operator<LA::MPI::Vector>(preconditioner_M) * op_B;
   //
   // IterationNumberControl   iteration_number_control_aS(30, 1.e-18);
   // SolverCG<Vector<double>> solver_aS(iteration_number_control_aS);
   //
   // const auto preconditioner_S =
   //    inverse_operator(op_aS, solver_aS, PreconditionIdentity());
   //
   // const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;
   LA::MPI::Vector tmp;
   tmp.reinit(F);
   op_M_inv.vmult(tmp, F);
   LA::MPI::Vector schur_rhs;
   schur_rhs.reinit(G);
   B.Tvmult(schur_rhs, tmp);
   schur_rhs -= G;
   //
   // SolverControl            solver_control_S(6000, 1.e-8);
   // SolverCG<Vector<double>> solver_S(solver_control_S);
   //
   // const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
   //
   timer.start();
   // P = op_S_inv * schur_rhs;
  SolverControl solver_control_S(3000, 1e-8);
   // SchurMatrixInverse<MassMatrixInverse, SchurMatrixInverse<LA::PreconditionJacobi, PreconditionIdentity, IterationNumberControl>, SolverControl> op_S_inv(op_S, op_aS_inv, solver_control_S);
   SchurMatrixInverse op_S_inv(op_S, op_aS_inv, solver_control_S);
   op_S_inv.vmult(P, schur_rhs);
   timer.stop();
   // op_M_inv.vmult(P, P);
   phi_iteration = op_S_inv.solve_iter;
   phi_time = timer.last_wall_time();
   // phi_iteration = solver_control_S.last_step();
   // phi_iteration = 0;

   timer.start();
   // U = op_M_inv * (F - op_B * P);
   LA::MPI::Vector tmp_1;
   tmp_1.reinit(U);
   B.vmult(tmp_1, P);
   tmp_1 -= F;
   tmp_1 *= -1;
   op_M_inv.vmult(U,tmp_1);
   constraints.distribute(distributed_solution);
   block_vars.solution = distributed_solution;
   timer.stop();
   j_iteration = reduction_control.last_step();
   // j_iteration = op_M_inv.solve_iter;
   j_time = timer.last_wall_time();
   
}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::solve_umfpack(int&    phi_iteration,
                                        int&    j_iteration,
                                        double& phi_time,
                                        double& j_time)
{
   timer.start();
   SolverControl cn;
   LA::SparseDirectMUMPS solver(cn);
   // solver.initialize(vars.system_matrix);
   LA::MPI::Vector distributed_solution(dof_handler.locally_owned_dofs(),
                                mpi_comm);
   solver.solve(vars.system_matrix, distributed_solution, vars.system_rhs);
   constraints.distribute(distributed_solution);
   vars.solution = distributed_solution;
   // */
   timer.stop();

   phi_iteration = 0;
   j_iteration = 0;
   phi_time = timer.last_wall_time();
   j_time = 0.0;
}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::solve_petsc_gmres(int&    phi_iteration,
                                      int&    j_iteration,
                                      double& phi_time,
                                      double& j_time)
{
   timer.start();
   /*SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
   LA::SolverGMRES::AdditionalData additional_data(false, 30);
   LA::SolverGMRES solver(solver_control, additional_data);

   // TODO: Not able to use ILU with BlockMatrix/Vector
   SparseILU<double> preconditioner;
   preconditioner.initialize(system_matrix);

   solver.solve(system_matrix, solution, system_rhs, LA::PreconditionNone());*/

   auto A = vars.system_matrix.petsc_matrix();
   auto b = vars.system_rhs.petsc_vector();
   // auto x = vars.solution.petsc_vector();

   LA::MPI::Vector distributed_solution;
   distributed_solution.reinit(dof_handler.locally_owned_dofs(), mpi_comm);
   auto x = distributed_solution.petsc_vector();

   KSP ksp;
   PC  pc;
   AssertPETSc(KSPCreate(mpi_comm, &ksp));
   AssertPETSc(KSPSetType(ksp, KSPGMRES));
   AssertPETSc(KSPGetPC(ksp, &pc));
   // AssertPETSc(PCSetType(pc, PCJACOBI)); // works fine
   // AssertPETSc(PCSetType(pc, PCILU)); // gives same error
   // AssertPETSc(PCSetType(pc, PCILU));
   // AssertPETSc(PCSetType(pc, PCNONE));
   AssertPETSc(PCFactorReorderForNonzeroDiagonal(pc, 1e-6));
   AssertPETSc(PCSetType(pc, PCHYPRE));
   AssertPETSc(PCHYPRESetType(pc, "euclid"));
   // AssertPETSc(PCHYPRESetType(pc, "ilu"));
   // AssertPETSc(PCHYPRESetType(pc, "pilut"));

   // AssertPETSc(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));
   // AssertPETSc(PetscOptionsSetValue(nullptr, "-pc_hypre_euclid_level", "0")); 
   // AssertPETSc(PetscOptionsSetValue(nullptr, "-pc_hypre_ilu_level", "0")); 
   AssertPETSc(KSPSetOperators(ksp, A, A));
   AssertPETSc(KSPSetTolerances(ksp, 1e-6, PETSC_CURRENT, PETSC_CURRENT, 4000));
   AssertPETSc(KSPGMRESSetRestart(ksp, 30));
   AssertPETSc(KSPSetFromOptions(ksp));
   AssertPETSc(KSPSetUp(ksp));
   AssertPETSc(KSPSolve(ksp, b, x));
   AssertPETSc(KSPGetIterationNumber(ksp, &phi_iteration));
   AssertPETSc(KSPDestroy(&ksp));
   timer.stop();

   constraints.distribute(distributed_solution);
   vars.solution = distributed_solution;

   j_iteration = 0;
   phi_time = timer.last_wall_time();
   j_time = 0.0;

}
//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::solve_gmres(int&    phi_iteration,
                    int&    j_iteration,
                    double& phi_time,
                    double& j_time)
{
   timer.start();
   // SolverControl solver_control(3000, 1e-6 * vars.system_rhs.l2_norm());
   // LA::SolverGMRES solver(solver_control);
   // LA::MPI::Vector distributed_solution(dof_handler.locally_owned_dofs(),
   //                              mpi_comm);
   //
   // // TODO: Not able to use ILU with BlockMatrix/Vector
   // // SparseILU<LA::MPI::SparseMatrix> preconditioner;
   // //preconditioner.initialize(system_matrix);
   // // LA::PreconditionILU preconditioner;               //    not working cause matrix has diagonal entries as 0
   // // LA::PreconditionLU preconditioner;                //    not working cause matrix has diagonal entries as 0
   // LA::PreconditionJacobi preconditioner;            //    working
   // // LA::PreconditionBoomerAMG preconditioner;         //    no convergence
   // // LA::PreconditionICC preconditioner;               //    not working cause matrix has diagonal entries as 0
   // // LA::PreconditionParaSails preconditioner;         //    not working cause matrix not SPD
   // // LA::PreconditionSSOR preconditioner;              //    no convergence is very slow
   // // LA::PreconditionBDDC<dim> preconditioner;
   // // LA::PreconditionBlockJacobi preconditioner;       //    not working cause matrix has diagonal entries as 0 
   // // LA::PreconditionSOR preconditioner;               //    no convergence is very slow
   // // LA::PreconditionNone preconditioner;
   // preconditioner.initialize(vars.system_matrix);

   // solver.solve(vars.system_matrix, vars.solution, vars.system_rhs, preconditioner);
   // solver.solve(vars.system_matrix, vars.solution, vars.system_rhs, preconditioner);
   // solver.solve(vars.system_matrix, distributed_solution, vars.system_rhs, preconditioner);
   // constraints.distribute(distributed_solution);
   // vars.solution = distributed_solution;

   // pcout << 1e-4* block_vars.system_rhs.l2_norm() <<std::endl;
   SolverControl solver_control(3000, 1e-8 * block_vars.system_rhs.l2_norm());
   SolverGMRES<LA::MPI::BlockVector> solver(solver_control);


   const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
   const unsigned int n_c = dofs_per_component[0],
                      n_p = dofs_per_component[dim];

   const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
   // IndexSet locally_relevant_dofs =
   //     DoFTools::extract_locally_relevant_dofs(dof_handler);

   std::vector<IndexSet>
       owned_partitioning = {locally_owned_dofs.get_view(0, n_c),
                             locally_owned_dofs.get_view(n_c, n_c + n_p)};
   // std::vector<IndexSet>
   //    relevant_partitioning = {locally_relevant_dofs.get_view(0, n_c),
   //                             locally_relevant_dofs.get_view(n_c, n_c + n_p)};
   LA::MPI::BlockVector distributed_solution(owned_partitioning, mpi_comm);

   // Preconditioner
   const auto& M = block_vars.system_matrix.block(0, 0);
   const auto& B = block_vars.system_matrix.block(0, 1);

   const auto & U = block_vars.system_rhs.block(0);

   // LA::PreconditionJacobi preconditioner_M;
   LA::PreconditionLU preconditioner_M;
   preconditioner_M.initialize(M);
   
   //--------------------------------------------------------------------------------------
   // LA::MPI::Vector tmp(U);
   // LA::MPI::Vector tmp_2(U);
   // LA::MPI::Vector tmp_3(U);
   // tmp = 0;
   // tmp(0) = 1;
   // tmp(1) = 0.5;
   // tmp.compress(VectorOperation::insert);
   // M.vmult(tmp_2, tmp);
   // pcout << tmp.l2_norm() << std::endl;
   // preconditioner_M.vmult(tmp_3, tmp_2);
   // pcout << "hllo" << std::endl;
   // tmp_3.add(-1, tmp);
   // double tmp_1 = tmp_3.l2_norm() / tmp.l2_norm();
   // pcout << tmp_1 << std::endl;
   //--------------------------------------------------------------------------------------


   // const auto &U = block_vars.system_rhs.block(0);

   // SchurMatrix<LA::PreconditionJacobi> op_aS(preconditioner_M, B, U);
   SchurMatrix<LA::PreconditionLU> op_aS(preconditioner_M, B, U);

  PreconditionIdentity identity;
  // int iterations = 30;
  // IterationNumberControl iterations(30, 1e-18);
   SolverControl iterations(3000, 1e-7);
  SchurMatrixInverse op_aS_inv(op_aS, identity, iterations);

   // BlockSchurPreconditioner<SchurMatrixInverse<SchurMatrix<LA::PreconditionJacobi>, PreconditionIdentity, IterationNumberControl>, LA::PreconditionJacobi> bsp(block_vars.system_matrix.block(0,1), preconditioner_M, op_aS_inv);
   BlockSchurPreconditioner bsp(block_vars.system_matrix.block(0,1), preconditioner_M, op_aS_inv);
   solver.solve(block_vars.system_matrix, distributed_solution, block_vars.system_rhs, bsp);
   // solver.solve(block_vars.system_matrix, distributed_solution, block_vars.system_rhs, PreconditionIdentity());
   constraints.distribute(distributed_solution);
   block_vars.solution = distributed_solution;
   timer.stop();

   phi_iteration = solver_control.last_step();
   j_iteration = 0;
   phi_time = timer.last_wall_time();
   j_time = 0.0;
}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::solve(int&    phi_iteration,
                                int&    j_iteration,
                                double& phi_time,
                                double& j_time)
{
   if(linear_solver == "schur")
      solve_schur(phi_iteration, j_iteration, phi_time, j_time);
   else if(linear_solver == "umfpack")
      solve_umfpack(phi_iteration, j_iteration, phi_time, j_time);
   else if(linear_solver == "petsc_gmres")
       solve_petsc_gmres(phi_iteration, j_iteration, phi_time, j_time);
   else
      solve_gmres(phi_iteration, j_iteration, phi_time, j_time);

   pcout << "Time to solve (phi,j,total) = " << phi_time << ", " << j_time
             << ", " << phi_time + j_time << std::endl;
}

//------------------------------------------------------------------------------
template <int dim>
template<typename VariableStruct>
void
MixedLaplaceProblem<dim>::compute_errors(double& phi_err,
                                         double& j_err,
                                         double& d_err,
                                         VariableStruct &VarStruct)
{
   PrescribedSolution::ExactSolution<dim> exact_solution;
   const QGauss<dim> quadrature(degree + 2);

       {
          const ComponentSelectFunction<dim> scalar_mask(dim, dim + 1);
          Vector<double> cellwise_errors(triangulation.n_active_cells());
          VectorTools::integrate_difference(dof_handler,
                                            VarStruct.solution,
                                            exact_solution,
                                            cellwise_errors,
                                            quadrature,
                                            VectorTools::L2_norm,
                                            &scalar_mask);
          phi_err = VectorTools::compute_global_error(triangulation,
                                                      cellwise_errors,
                                                      VectorTools::L2_norm);
       }

       {
          const ComponentSelectFunction<dim> vector_mask(std::make_pair(0, dim),
                                                         dim + 1);
          Vector<double> cellwise_errors(triangulation.n_active_cells());
          VectorTools::integrate_difference(dof_handler,
                                            VarStruct.solution,
                                            exact_solution,
                                            cellwise_errors,
                                            quadrature,
                                            VectorTools::L2_norm,
                                            &vector_mask);
          j_err = VectorTools::compute_global_error(triangulation,
                                                    cellwise_errors,
                                                    VectorTools::L2_norm);
       }

       {
          const ComponentSelectFunction<dim> vector_mask(std::make_pair(0, dim),
                                                         dim + 1);
          Vector<double> cellwise_errors(triangulation.n_active_cells());
          VectorTools::integrate_difference(dof_handler,
                                            VarStruct.solution,
                                            exact_solution,
                                            cellwise_errors,
                                            quadrature,
                                            VectorTools::Hdiv_seminorm,
                                            &vector_mask);
          d_err = VectorTools::compute_global_error(triangulation,
                                                    cellwise_errors,
                                                    VectorTools::L2_norm);
       }
}

//------------------------------------------------------------------------------
// TODO
template <int dim>
template<typename VariableStruct>
void
MixedLaplaceProblem<dim>::output_results(VariableStruct &VarStruct)
{
   timer.start();
   static int step = 0;
   static std::vector<std::pair<double, std::string>> pvtu_files;
   const unsigned int n_digits_for_counter = 2;

   std::vector<std::string> solution_names(dim, "vector");
   solution_names.emplace_back("scalar");
   std::vector<DataComponentInterpretation::DataComponentInterpretation>
   interpretation(dim,
                  DataComponentInterpretation::component_is_part_of_vector);
   interpretation.push_back(DataComponentInterpretation::component_is_scalar);

   DataOut<dim> data_out;
       data_out.add_data_vector(dof_handler,
                                VarStruct.solution,
                                solution_names,
                                interpretation);

       Postprocessor<dim> div_j;
       data_out.add_data_vector(dof_handler, VarStruct.solution, div_j);
   data_out.build_patches(degree + 1);
   // std::ofstream output("solution_" + Utilities::int_to_string(n) + ".vtu");
   // data_out.write_vtu(output);
   data_out.write_vtu_with_pvtu_record("./", "sol", step, mpi_comm, n_digits_for_counter);
   std::string fname = "sol_" + Utilities::int_to_string(step, n_digits_for_counter) + ".pvtu";
   pvtu_files.emplace_back(step, fname);
   std::ofstream pvd_file("sol.pvd");
   DataOutBase::write_pvd_record(pvd_file, pvtu_files);
   ++step;

   timer.stop();
   // double time_elapsed = timer.last_wall_time();
}

//------------------------------------------------------------------------------
template <int dim>
void
MixedLaplaceProblem<dim>::run(std::vector<int>&    ncell,
                              std::vector<double>& h_array,
                              std::vector<int>&    phi_dofs,
                              std::vector<int>&    j_dofs,
                              std::vector<double>& phi_error,
                              std::vector<double>& j_error,
                              std::vector<double>& d_error,
                              std::vector<int>&    phi_iterations,
                              std::vector<int>&    j_iterations,
                              std::vector<double>& phi_time,
                              std::vector<double>& j_time)
{
   for(unsigned int i = 0; i < nrefine; ++i)
   {
      timer.reset();

      make_grid(i + initial_refine);
      pcout << "---------------Grid level = " << i << "-----------------\n";
      setup_system();

      // Solve J,phi
      if(linear_solver == "schur" || linear_solver == "gmres")
      {
         assemble_system<BlockVariables>(block_vars);
      } else {
         assemble_system<Variables>(vars);
      }

      solve(phi_iterations[i], j_iterations[i], phi_time[i], j_time[i]);

      if(linear_solver == "schur" || linear_solver == "gmres")
      {
          compute_errors<BlockVariables>(phi_error[i], j_error[i], d_error[i], block_vars);
          output_results<BlockVariables>(block_vars);
      } else {
          compute_errors<Variables>(phi_error[i], j_error[i], d_error[i], vars);
          output_results<Variables>(vars);
      }


      ncell[i] = triangulation.n_active_cells();
      const std::vector<types::global_dof_index> dofs_per_component =
         DoFTools::count_dofs_per_fe_component(dof_handler);
      const unsigned int n_c = dofs_per_component[0],
                         n_p = dofs_per_component[dim];
      phi_dofs[i] = n_p;
      j_dofs[i] = n_c;
      h_array[i] = h_max;
   }
}

//------------------------------------------------------------------------------
int
main(int argc, char **argv)
{
   Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
   try
   {
   const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if(rank == 0)   std::cout << "Solving " << problem << " problem\n";

    ParameterHandler prm;
    ParameterReader param(prm);
    param.read_parameters("input.prm");

    unsigned int degree = prm.get_integer("degree");
    unsigned int nrefine = prm.get_integer("nrefine");
    unsigned int initial_refine = prm.get_integer("initial_refine");
    if (rank == 0)std::cout << "Using FE = RT(" << degree << "), DG(" << degree << ")\n";

    std::vector<int> ncell(nrefine),  p_dofs(nrefine), j_dofs(nrefine),
                     p_iterations(nrefine), j_iterations(nrefine);
    std::vector<double> p_error(nrefine),  j_error(nrefine), d_error(nrefine),
                        p_time(nrefine), j_time(nrefine), h_array(nrefine);

    MixedLaplaceProblem<2> mixed_laplace_problem(nrefine,
                                                 degree,
                                                 initial_refine,
                                                 prm.get_double("perturb grid"),
                                                 prm.get("solver"));
    mixed_laplace_problem.run(ncell,
                              h_array,
                              p_dofs,
                              j_dofs,
                              p_error,
                              j_error,
                              d_error,
                              p_iterations,
                              j_iterations,
                              p_time,
                              j_time);

      ConvergenceTable  convergence_table;
      for(unsigned int n = 0; n < nrefine; ++n)
      {
         convergence_table.add_value("cells", ncell[n]);
         convergence_table.add_value("dofs(p)",  p_dofs[n]);
         convergence_table.add_value("dofs(j)",  j_dofs[n]);
         convergence_table.add_value("hmax", h_array[n]);
         convergence_table.add_value("time(p)", p_time[n]);
         convergence_table.add_value("iter(p)", p_iterations[n]);
         convergence_table.add_value("error(p)", p_error[n]);
         convergence_table.add_value("time(j)", j_time[n]);
         convergence_table.add_value("iter(j)", j_iterations[n]);
         convergence_table.add_value("error(j)", j_error[n]);
         convergence_table.add_value("error(d)", d_error[n]);
      }

      convergence_table.set_precision("error(j)", 3);
      convergence_table.set_scientific("error(j)", true);
      convergence_table.set_precision("error(p)", 3);
      convergence_table.set_scientific("error(p)", true);
      convergence_table.set_precision("error(d)", 3);
      convergence_table.set_scientific("error(d)", true);

      convergence_table.set_tex_caption("cells", "cells");
      convergence_table.set_tex_caption("dofs(p)", "dofs(p)");
      convergence_table.set_tex_caption("dofs(j)", "dofs(j)");
      convergence_table.set_tex_caption("hmax", "hmax");
      convergence_table.set_tex_caption("time(p)", "time(p)");
      convergence_table.set_tex_caption("iter(p)", "iter(p)");
      convergence_table.set_tex_caption("error(p)", "error(p)");
      convergence_table.set_tex_caption("time(j)", "time(j)");
      convergence_table.set_tex_caption("iter(j)", "iter(j)");
      convergence_table.set_tex_caption("error(j)", "error(j)");
      convergence_table.set_tex_caption("error(d)", "error(div)");

      convergence_table.set_tex_format("cells", "r");
      convergence_table.set_tex_format("dofs(p)", "r");
      convergence_table.set_tex_format("dofs(j)", "r");
      convergence_table.set_tex_format("hmax",  "r");
      convergence_table.set_tex_format("time(p)", "r");
      convergence_table.set_tex_format("iter(p)",  "r");
      convergence_table.set_tex_format("time(j)", "r");
      convergence_table.set_tex_format("iter(j)",  "r");

      convergence_table.evaluate_convergence_rates
      ("error(j)", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates
      ("error(p)", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates
      ("error(d)", ConvergenceTable::reduction_rate_log2);

      if(rank == 0){std::cout << std::endl;
      convergence_table.write_text(std::cout);

      std::ofstream tex_file(problem + ".tex");
      convergence_table.write_tex(tex_file);
      std::ofstream text_file(problem + ".txt");
      convergence_table.write_text(text_file);
    }
   }

   catch(std::exception& exc)
   {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
   }
   catch(...)
   {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
   }

   return 0;
}
