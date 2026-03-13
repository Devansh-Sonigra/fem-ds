namespace PrescribedSolution
{
   const double pi = M_PI;

   //---------------------------------------------------------------------------
   template <int dim>
   class ExactSolution : public Function<dim>
   {
   public:
      ExactSolution() : Function<dim>(dim+1) {}

      void vector_value(const Point<dim>& p,
                        Vector<double>&   value) const override;
      void vector_gradient(const Point<dim>&           p,
                           std::vector<Tensor<1,dim>>& value) const override;
   };

   template <>
   void ExactSolution<2>::vector_value(const Point<2>& p,
                                       Vector<double>& value) const
   {
      const double x = p[0];
      const double y = p[1];
      const double b = 2 * pi;
      const double cx = cos(b * x);
      const double sx = sin(b * x);
      const double cy = cos(b * y);
      const double sy = sin(b * y);

      const double cc = cx * cy;
      const double cs = cx * sy;
      const double sc = sx * cy;
      const double ss = sx * sy;
      const double exp_xy = exp(x + y);

      // value[0] = 2 * pi * cx * sy;
      // value[1] = 2 * pi * sx * cy;
      // value[2] = sx * sy;

      // value[0] = (sx + b * x * cx) * y * sy;
      // value[1] = (sy + b * y * cy) * x * sx;
      // value[2] = x *  y * sx * sy;

      value[0] = exp_xy * (b * cs + ss); 
      value[1] = exp_xy * (b * sc + ss);
      value[2] = exp_xy * ss;
   }

   template <>
   void ExactSolution<2>::vector_gradient(const Point<2>&           p,
                                          std::vector<Tensor<1,2>>& value) const
   {
      const double x = p[0];
      const double y = p[1];
      const double b  = 2 * pi;
      const double cx = cos(b * x);
      const double sx = sin(b * x);
      const double cy = cos(b * y);
      const double sy = sin(b * y);
      const double a  = pow(2 * pi, 2);

      const double cc = cx * cy;
      const double cs = cx * sy;
      const double sc = sx * cy;
      const double ss = sx * sy;
      const double exp_xy = exp(x + y);
      // value[0][0] = -a * sx * sy;
      // value[0][1] =  a * cx * cy;
      //
      // value[1][0] =  a * cx * cy;
      // value[1][1] = -a * sx * sy;
      //
      // value[2][0] = b * cx * sy;
      // value[2][1] = b * cy * sx;

      // value[0][0] = b * (2 * cx  - b * x * sx) * y * sy;
      // value[0][1] = (sx + b * x * cx) * (sy + b * y * cy);
      //
      // // value[1][0] = (sx + b * x * cx) * (sy + b * y * cy);
      // value[1][0] = value[0][1];
      // value[1][1] = b * (2 * cy - b * y * sy) * x * sx;
      //
      // value[2][0] = (sx + b * x * cx) * y * sy;
      // value[2][1] = (sy + b * y * cy) * x * sx;

      value[0][0] = exp_xy * (( -b * b + 1) * ss + 2 * b * cs);
      value[0][1] = exp_xy * (b * b * cc + 2 * b * cs + ss);

      // value[1][0] = (sx + b * x * cx) * (sy + b * y * cy);
      value[1][0] = value[0][1];
      value[1][1] = exp_xy * ((-b * b + 1) * ss + 2 * b * sc);

      value[2][0] = exp_xy * (b * cs + ss);
      value[2][1] = exp_xy * (b * sc + ss);
   }

   //---------------------------------------------------------------------------
   //---------------------------------------------------------------------------
   template <int dim>
   class RHSFunction : public Function<dim>
   {
   public:
      RHSFunction() : Function<dim>() {}

      double value(const Point<dim> &p,
                   const unsigned int component = 0) const override;
   };

   template <>
   double RHSFunction<2>::value(const Point<2> &p,
                                const unsigned int /*component*/) const
   {
      // return 8 * pi * pi * sin(2 * pi * p[0]) * sin(2 * pi * p[1]);
      // return -2 * pi * ( (2 * cos(2 * pi * p[0]) - 2 * pi * p[0] * sin(2 * pi * p[0])) * p[1] * sin(2 * pi * p[1]) + (2 * cos(2 * pi * p[1]) - 2 * pi * p[1] * sin(2 * pi * p[1])) * p[0] * sin(2 * pi * p[0]));
      const double x = p[0];
      const double y = p[1];
      const double b  = 2 * pi;
      const double cx = cos(b * x);
      const double sx = sin(b * x);
      const double cy = cos(b * y);
      const double sy = sin(b * y);
      const double a  = pow(2 * pi, 2);

      const double cc = cx * cy;
      const double cs = cx * sy;
      const double sc = sx * cy;
      const double ss = sx * sy;
      const double exp_xy = exp(x + y);

      return -(exp_xy * (( -b * b + 1) * ss + 2 * b * cs) + exp_xy * ((-b * b + 1) * ss + 2 * b * sc));
   }
} // namespace PrescribedSolution
