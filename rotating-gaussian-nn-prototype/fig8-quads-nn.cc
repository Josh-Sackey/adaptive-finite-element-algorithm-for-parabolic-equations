#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "neural_surrogate.h"

using namespace dealii;

namespace RotatingGaussian
{
  constexpr double width = 500.0;
  constexpr double radius = 0.3;
  constexpr double omega = 2.0 * numbers::PI;

  class ExactSolution : public Function<2>
  {
  public:
    ExactSolution(const double time = 0.0)
      : Function<2>(1, time)
    {}

    double value(const Point<2> &p,
                 const unsigned int = 0) const override
    {
      const double cx = radius * std::cos(omega * this->get_time());
      const double cy = radius * std::sin(omega * this->get_time());
      return std::exp(-width * ((p[0] - cx) * (p[0] - cx) +
                                (p[1] - cy) * (p[1] - cy)));
    }

    Tensor<1, 2> gradient(const Point<2> &p,
                          const unsigned int = 0) const override
    {
      const double cx = radius * std::cos(omega * this->get_time());
      const double cy = radius * std::sin(omega * this->get_time());
      Tensor<1, 2> result;
      result[0] = -2.0 * width * (p[0] - cx) * value(p);
      result[1] = -2.0 * width * (p[1] - cy) * value(p);
      return result;
    }
  };

  class RightHandSide : public Function<2>
  {
  public:
    RightHandSide(const double time = 0.0)
      : Function<2>(1, time)
    {}

    double value(const Point<2> &p,
                 const unsigned int = 0) const override
    {
      const double t = this->get_time();
      const double cx = radius * std::cos(omega * t);
      const double cy = radius * std::sin(omega * t);
      const double cpx = -radius * omega * std::sin(omega * t);
      const double cpy = radius * omega * std::cos(omega * t);
      const double dx = p[0] - cx;
      const double dy = p[1] - cy;
      const double u = std::exp(-width * (dx * dx + dy * dy));
      const double u_t = 2.0 * width * (dx * cpx + dy * cpy) * u;
      const double laplace_u =
        (4.0 * width * width * (dx * dx + dy * dy) - 4.0 * width) * u;
      return u_t - laplace_u;
    }
  };

  class Problem
  {
  public:
    Problem(const double top_fraction, const double final_time);
    void run();

  private:
    void setup_system();
    void assemble_initial_projection();
    void assemble_time_step(const double time);
    void solve();
    Vector<float> recovery_indicators() const;
    void adapt_mesh();
    void refine_without_transfer(const Vector<float> &indicators);
    TrainingResult train_surrogate(const double time);
    double exact_gradient_norm(const double time) const;
    void report_errors(const std::string &phase,
                       const unsigned int index,
                       const double time,
                       const bool print_to_console = true);
    void output_results(const unsigned int output_index,
                        const double time,
                        const Vector<float> &indicator);

    Triangulation<2> triangulation;
    FE_Q<2> fe;
    DoFHandler<2> dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;
    std::vector<std::pair<double, std::string>> pvd_records;
    std::ofstream history_file;
    std::ofstream training_file;
    NeuralSurrogate surrogate;

    const double time_step = 0.01;
    const double final_time;
    const double top_fraction;
    const double bottom_fraction;
    const unsigned int min_grid_level = 0;
    const unsigned int max_grid_level = 5;
    const unsigned int max_adaptive_iterations = 7;
    const double estimator_tolerance = 0.01;
  };

  Problem::Problem(const double requested_top_fraction,
                   const double requested_final_time)
    : fe(1)
    , dof_handler(triangulation)
    , final_time(requested_final_time)
    , top_fraction(requested_top_fraction)
    , bottom_fraction(1.0 - requested_top_fraction)
  {}

  void Problem::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  void Problem::assemble_initial_projection()
  {
    system_matrix = 0;
    system_rhs = 0;
    QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe,
                          quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);
    FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);
    ExactSolution exact(0.0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              cell_rhs(i) += fe_values.shape_value(i, q) *
                             exact.value(fe_values.quadrature_point(q)) *
                             fe_values.JxW(q);
              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                     fe_values.shape_value(j, q) *
                                     fe_values.JxW(q);
            }
        cell->get_dof_indices(local_dofs);
        constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dofs,
                                                system_matrix,
                                                system_rhs);
      }

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             exact,
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }

  void Problem::assemble_time_step(const double time)
  {
    system_matrix = 0;
    system_rhs = 0;
    QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);
    std::vector<double> previous_values(quadrature.size());
    RightHandSide rhs(time);
    ExactSolution exact(time);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        std::vector<std::array<double, 2>> points(quadrature.size());
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          points[q] = {fe_values.quadrature_point(q)[0],
                       fe_values.quadrature_point(q)[1]};
        previous_values = surrogate.values(points);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              cell_rhs(i) += fe_values.shape_value(i, q) *
                             (previous_values[q] + time_step *
                                                rhs.value(fe_values.quadrature_point(q))) *
                             fe_values.JxW(q);
              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) +
                   time_step * fe_values.shape_grad(i, q) *
                     fe_values.shape_grad(j, q)) *
                  fe_values.JxW(q);
            }
        cell->get_dof_indices(local_dofs);
        constraints.distribute_local_to_global(cell_matrix,
                                                cell_rhs,
                                                local_dofs,
                                                system_matrix,
                                                system_rhs);
      }

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             exact,
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }

  void Problem::solve()
  {
    SolverControl control(std::max<unsigned int>(1000, dof_handler.n_dofs()),
                          1e-12 * system_rhs.l2_norm() + 1e-14);
    SolverCG<Vector<double>> cg(control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
  }

  Vector<float> Problem::recovery_indicators() const
  {
    Vector<float> indicators(triangulation.n_active_cells());
    std::vector<Tensor<1, 2>> recovered(dof_handler.n_dofs());
    std::vector<unsigned int> counts(dof_handler.n_dofs(), 0);
    QTrapezoid<2> vertex_quadrature;
    FEValues<2> vertex_values(fe, vertex_quadrature, update_gradients);
    std::vector<Tensor<1, 2>> vertex_gradients(vertex_quadrature.size());
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        vertex_values.reinit(cell);
        vertex_values.get_function_gradients(solution, vertex_gradients);
        cell->get_dof_indices(local_dofs);
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
          {
            const unsigned int local = fe.component_to_system_index(0, v);
            recovered[local_dofs[local]] += vertex_gradients[v];
            ++counts[local_dofs[local]];
          }
      }
    for (unsigned int i = 0; i < recovered.size(); ++i)
      if (counts[i] != 0)
        recovered[i] /= static_cast<double>(counts[i]);

    QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);
    std::vector<Tensor<1, 2>> gradients(quadrature.size());
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(solution, gradients);
        cell->get_dof_indices(local_dofs);
        double eta_squared = 0.0;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            Tensor<1, 2> recovered_q;
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              recovered_q += fe_values.shape_value(i, q) * recovered[local_dofs[i]];
            eta_squared += (recovered_q - gradients[q]).norm_square() *
                           fe_values.JxW(q);
          }
        indicators[cell->active_cell_index()] = std::sqrt(eta_squared);
      }
    return indicators;
  }

  void Problem::adapt_mesh()
  {
    const Vector<float> indicators = recovery_indicators();
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                       indicators,
                                                       top_fraction,
                                                       bottom_fraction);

    // These are the same level guards used by Step-26. The initial 8x8 mesh
    // is level zero; five adaptive pre-refinements permit levels zero to five.
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<2, Vector<double>> transfer(dof_handler);
    triangulation.prepare_coarsening_and_refinement();
    transfer.prepare_for_coarsening_and_refinement(solution);
    triangulation.execute_coarsening_and_refinement();
    setup_system();
    transfer.interpolate(old_solution);
    constraints.distribute(old_solution);
    solution = old_solution;
  }

  void Problem::refine_without_transfer(const Vector<float> &indicators)
  {
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                       indicators,
                                                       top_fraction,
                                                       0.0);
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    triangulation.execute_coarsening_and_refinement();
    setup_system();
  }

  TrainingResult Problem::train_surrogate(const double time)
  {
    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
                                         dof_handler,
                                         support_points);
    std::vector<std::array<double, 2>> points(dof_handler.n_dofs());
    std::vector<double> values(dof_handler.n_dofs());
    for (const auto &entry : support_points)
      {
        points[entry.first] = {entry.second[0], entry.second[1]};
        values[entry.first] = solution[entry.first];
      }
    const unsigned int iterations = surrogate.is_trained() ? 80 : 400;
    const TrainingResult result = surrogate.train(points, values, iterations);
    training_file << std::fixed << std::setprecision(8) << time << ','
                  << dof_handler.n_dofs() << ',' << result.closure_evaluations
                  << ',' << std::scientific << std::setprecision(12)
                  << result.final_loss << ',' << std::fixed
                  << std::setprecision(6) << result.seconds << '\n';
    training_file.flush();
    std::cout << "NN t=" << std::fixed << std::setprecision(2) << time
              << " samples=" << dof_handler.n_dofs()
              << " evaluations=" << result.closure_evaluations
              << std::scientific << " loss=" << result.final_loss
              << std::fixed << " seconds=" << result.seconds << std::endl;
    return result;
  }

  double Problem::exact_gradient_norm(const double time) const
  {
    ExactSolution exact(time);
    Vector<double> zero(dof_handler.n_dofs());
    Vector<float> difference(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      zero,
                                      exact,
                                      difference,
                                      QGauss<2>(fe.degree + 3),
                                      VectorTools::H1_seminorm);
    return VectorTools::compute_global_error(triangulation,
                                              difference,
                                              VectorTools::H1_seminorm);
  }

  void Problem::report_errors(const std::string &phase,
                              const unsigned int index,
                              const double time,
                              const bool print_to_console)
  {
    ExactSolution exact(time);
    Vector<float> difference(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact,
                                      difference,
                                      QGauss<2>(fe.degree + 3),
                                      VectorTools::L2_norm);
    const double l2 = VectorTools::compute_global_error(triangulation,
                                                        difference,
                                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact,
                                      difference,
                                      QGauss<2>(fe.degree + 3),
                                      VectorTools::H1_seminorm);
    const double h1 = VectorTools::compute_global_error(
      triangulation, difference, VectorTools::H1_seminorm);
    if (print_to_console)
      std::cout << phase << ' ' << std::setw(3) << index << " t=" << std::fixed
                << std::setprecision(2) << time << " cells="
                << triangulation.n_active_cells() << " dofs="
                << dof_handler.n_dofs() << std::scientific
                << std::setprecision(6) << " L2=" << l2 << " H1=" << h1
                << std::endl;

    history_file << phase << ',' << index << ',' << std::fixed
                 << std::setprecision(8) << time << ',' << top_fraction << ','
                 << bottom_fraction << ',' << triangulation.n_active_cells()
                 << ',' << dof_handler.n_dofs() << ',' << std::scientific
                 << std::setprecision(12) << l2 << ',' << h1 << '\n';
    history_file.flush();
  }

  void Problem::output_results(const unsigned int output_index,
                               const double time,
                               const Vector<float> &indicator)
  {
    Vector<double> exact_values(dof_handler.n_dofs());
    ExactSolution exact(time);
    VectorTools::interpolate(dof_handler, exact, exact_values);
    Vector<float> levels(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      levels[cell->active_cell_index()] = cell->level();

    DataOut<2> out;
    out.attach_dof_handler(dof_handler);
    out.add_data_vector(solution, "solution");
    out.add_data_vector(exact_values, "exact_solution");
    out.add_data_vector(indicator, "error_indicator");
    out.add_data_vector(levels, "refinement_level");
    out.build_patches();
    const std::string filename = "rotating-gaussian-" +
                                 Utilities::int_to_string(output_index, 4) + ".vtu";
    std::ofstream file(filename);
    out.write_vtu(file);
    pvd_records.emplace_back(time, filename);
    std::ofstream pvd("rotating-gaussian.pvd");
    DataOutBase::write_pvd_record(pvd, pvd_records);
  }

  void Problem::run()
  {
    history_file.open("simulation-history.csv");
    AssertThrow(history_file, ExcMessage("Could not open simulation-history.csv"));
    history_file << "phase,index,time,top_fraction,bottom_fraction,"
                    "active_cells,degrees_of_freedom,l2_error,h1_seminorm\n";
    training_file.open("neural-training.csv");
    AssertThrow(training_file, ExcMessage("Could not open neural-training.csv"));
    training_file << "time,samples,closure_evaluations,final_mse,seconds\n";

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                               {8, 8},
                                               Point<2>(-1.0, -1.0),
                                               Point<2>(1.0, 1.0));

    // Five complete solve/estimate/adapt cycles at t=0, followed by a final
    // projection solve on the fifth adapted mesh.
    for (unsigned int cycle = 0; cycle < 5; ++cycle)
      {
        setup_system();
        assemble_initial_projection();
        solve();
        report_errors("initial_adaptation", cycle + 1, 0.0);
        output_results(cycle, 0.0, recovery_indicators());
        adapt_mesh();
      }

    assemble_initial_projection();
    solve();
    report_errors("initial_adapted_mesh", 5, 0.0);
    output_results(5, 0.0, recovery_indicators());
    train_surrogate(0.0);

    unsigned int output_index = 6;
    const unsigned int n_steps = static_cast<unsigned int>(final_time / time_step);
    for (unsigned int step = 1; step <= n_steps; ++step)
      {
        const double time = step * time_step;
        dof_handler.clear();
        triangulation.clear();
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                   {8, 8},
                                                   Point<2>(-1.0, -1.0),
                                                   Point<2>(1.0, 1.0));
        setup_system();

        Vector<float> indicators;
        unsigned int adaptive_iteration = 0;
        for (; adaptive_iteration < max_adaptive_iterations;
             ++adaptive_iteration)
          {
            assemble_time_step(time);
            solve();
            indicators = recovery_indicators();
            const double relative_estimator =
              indicators.l2_norm() / exact_gradient_norm(time);
            std::cout << "t=" << std::fixed << std::setprecision(2) << time
                      << " adapt=" << adaptive_iteration + 1
                      << " cells=" << triangulation.n_active_cells()
                      << " dofs=" << dof_handler.n_dofs()
                      << std::scientific << " rel_eta=" << relative_estimator
                      << std::endl;
            if (relative_estimator <= estimator_tolerance ||
                adaptive_iteration + 1 == max_adaptive_iterations)
              break;
            refine_without_transfer(indicators);
          }
        // Preserve the paper's dt=0.01 and write every computed time level so
        // that the ParaView animation does not skip four solutions per frame.
        output_results(output_index++, time, indicators);
        report_errors("time_step", step, time, step % 5 == 0);
        train_surrogate(time);
      }
  }
} // namespace RotatingGaussian

int main(const int argc, char *argv[])
{
  try
    {
      const double top_fraction = argc > 1 ? std::stod(argv[1]) : 0.6;
      const double final_time = argc > 2 ? std::stod(argv[2]) : 1.0;
      AssertThrow(top_fraction > 0.0 && top_fraction < 1.0,
                  ExcMessage("top_fraction must be strictly between 0 and 1"));
      AssertThrow(final_time > 0.0 && final_time <= 1.0,
                  ExcMessage("final_time must be in (0,1]"));
      RotatingGaussian::Problem problem(top_fraction, final_time);
      problem.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << "ERROR: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
