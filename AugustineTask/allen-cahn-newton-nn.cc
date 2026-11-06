#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "neural_surrogate.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

namespace AllenCahn
{
  constexpr double epsilon = 0.01;
  constexpr double default_time_step = 0.0003;
  constexpr double default_final_time = 0.03;
  constexpr double boundary_value = -1.0;

  class InitialCondition : public Function<2>
  {
  public:
    double value(const Point<2> &p,
                 const unsigned int = 0) const override
    {
      const double dx = p[0] - 0.5;
      const double dy = p[1] - 0.5;
      const double radius = std::sqrt(dx * dx + dy * dy);
      return std::tanh((0.25 - radius) / (std::sqrt(2.0) * epsilon));
    }
  };

  class BoundaryValues : public Function<2>
  {
  public:
    double value(const Point<2> &,
                 const unsigned int = 0) const override
    {
      return boundary_value;
    }
  };

  class Problem
  {
  public:
    Problem(double final_time,
            std::string output_directory,
            unsigned int max_adaptive_iterations,
            double estimator_tolerance,
            unsigned int initial_subdivisions,
            unsigned int max_grid_level,
            unsigned int initial_adaptations,
            double time_step);
    void run();

  private:
    void make_initial_mesh();
    void setup_system();
    void set_initial_condition();
    void initialize_newton_guess_from_surrogate();
    double assemble_newton_system(bool assemble_matrix);
    unsigned int solve_newton(unsigned int time_index, double time);
    Vector<float> recovery_indicators() const;
    void mark_and_adapt(const Vector<float> &indicators);
    TrainingResult train_surrogate(double time);
    void update_nn_previous_vector();
    double compute_mass() const;
    double compute_energy() const;
    std::pair<double, double> solution_range() const;
    void output_results(unsigned int output_index,
                        double time,
                        const Vector<float> &indicators);
    void write_pvd() const;

    Triangulation<2> triangulation;
    FE_Q<2> fe;
    DoFHandler<2> dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> solution_update;
    Vector<double> system_rhs;
    Vector<double> nn_previous;

    NeuralSurrogate surrogate;
    std::vector<std::pair<double, std::string>> pvd_records;
    std::ofstream history_file;
    std::ofstream newton_file;
    std::ofstream training_file;
    std::chrono::steady_clock::time_point run_start;

    const double final_time;
    const std::string output_directory;
    const unsigned int max_adaptive_iterations;
    const double estimator_tolerance;
    const unsigned int initial_subdivisions;
    const unsigned int max_grid_level;
    const unsigned int initial_adaptations;
    const double time_step;
    const unsigned int max_newton_iterations = 20;
    const double newton_relative_tolerance = 1e-9;
    const double top_fraction = 0.6;
    const double bottom_fraction = 0.4;
  };

  Problem::Problem(const double requested_final_time,
                   std::string requested_output_directory,
                   const unsigned int requested_max_adaptive_iterations,
                   const double requested_estimator_tolerance,
                   const unsigned int requested_initial_subdivisions,
                   const unsigned int requested_max_grid_level,
                   const unsigned int requested_initial_adaptations,
                   const double requested_time_step)
    : fe(1)
    , dof_handler(triangulation)
    , final_time(requested_final_time)
    , output_directory(std::move(requested_output_directory))
    , max_adaptive_iterations(requested_max_adaptive_iterations)
    , estimator_tolerance(requested_estimator_tolerance)
    , initial_subdivisions(requested_initial_subdivisions)
    , max_grid_level(requested_max_grid_level)
    , initial_adaptations(requested_initial_adaptations)
    , time_step(requested_time_step)
  {}

  void Problem::make_initial_mesh()
  {
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                               {initial_subdivisions,
                                                initial_subdivisions},
                                               Point<2>(-1.0, -1.0),
                                               Point<2>(1.0, 1.0));
  }

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
    solution_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    nn_previous.reinit(dof_handler.n_dofs());
  }

  void Problem::set_initial_condition()
  {
    VectorTools::interpolate(dof_handler, InitialCondition(), solution);
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues(),
                                             boundary_values);
    for (const auto &entry : boundary_values)
      solution[entry.first] = entry.second;
    constraints.distribute(solution);
  }

  void Problem::initialize_newton_guess_from_surrogate()
  {
    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
                                         dof_handler,
                                         support_points);
    std::vector<std::array<double, 2>> points(dof_handler.n_dofs());
    for (const auto &entry : support_points)
      points[entry.first] = {entry.second[0], entry.second[1]};
    const auto values = surrogate.values(points);
    for (unsigned int i = 0; i < values.size(); ++i)
      solution[i] = values[i];
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues(),
                                             boundary_values);
    for (const auto &entry : boundary_values)
      solution[entry.first] = entry.second;
    constraints.distribute(solution);
    update_nn_previous_vector();
  }

  double Problem::assemble_newton_system(const bool assemble_matrix)
  {
    if (assemble_matrix)
      system_matrix = 0;
    system_rhs = 0;

    const QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe,
                          quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);
    std::vector<double> current_values(quadrature.size());
    std::vector<Tensor<1, 2>> current_gradients(quadrature.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, current_values);
        fe_values.get_function_gradients(solution, current_gradients);

        std::vector<std::array<double, 2>> points(quadrature.size());
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          points[q] = {fe_values.quadrature_point(q)[0],
                       fe_values.quadrature_point(q)[1]};
        const auto previous_values = surrogate.values(points);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const double u = current_values[q];
            const double reaction = (u - u * u * u) /
                                    (epsilon * epsilon);
            const double jacobian_reaction = (3.0 * u * u - 1.0) /
                                             (epsilon * epsilon);
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              {
                const double phi_i = fe_values.shape_value(i, q);
                const auto grad_i = fe_values.shape_grad(i, q);
                // -F(u_k): the right-hand side for the Newton correction.
                cell_rhs(i) +=
                  ((previous_values[q] - u) / time_step * phi_i -
                   current_gradients[q] * grad_i + reaction * phi_i) *
                  fe_values.JxW(q);
                if (assemble_matrix)
                  for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      (phi_i * fe_values.shape_value(j, q) / time_step +
                       grad_i * fe_values.shape_grad(j, q) +
                       jacobian_reaction * phi_i *
                         fe_values.shape_value(j, q)) *
                      fe_values.JxW(q);
              }
          }
        cell->get_dof_indices(local_dofs);
        if (assemble_matrix)
          constraints.distribute_local_to_global(cell_matrix,
                                                  cell_rhs,
                                                  local_dofs,
                                                  system_matrix,
                                                  system_rhs);
        else
          constraints.distribute_local_to_global(cell_rhs,
                                                  local_dofs,
                                                  system_rhs);
      }

    std::map<types::global_dof_index, double> zero_update;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<2>(),
                                             zero_update);
    if (assemble_matrix)
      MatrixTools::apply_boundary_values(zero_update,
                                         system_matrix,
                                         solution_update,
                                         system_rhs);
    else
      for (const auto &entry : zero_update)
        system_rhs[entry.first] = 0.0;
    return system_rhs.l2_norm();
  }

  unsigned int Problem::solve_newton(const unsigned int time_index,
                                     const double time)
  {
    double initial_residual = -1.0;
    for (unsigned int iteration = 0; iteration < max_newton_iterations;
         ++iteration)
      {
        solution_update = 0;
        const double residual = assemble_newton_system(true);
        if (iteration == 0)
          initial_residual = std::max(residual, 1e-30);
        const double relative_residual = residual / initial_residual;
        newton_file << time_index << ',' << std::setprecision(12) << time
                    << ',' << iteration << ',' << std::scientific << residual
                    << ',' << relative_residual << '\n';
        std::cout << "  Newton " << iteration << " residual="
                  << std::scientific << residual << " relative="
                  << relative_residual << std::endl;
        if (relative_residual <= newton_relative_tolerance || residual < 1e-11)
          return iteration;

        SparseDirectUMFPACK direct;
        direct.initialize(system_matrix);
        direct.vmult(solution_update, system_rhs);
        constraints.distribute(solution_update);
        solution += solution_update;
        constraints.distribute(solution);
      }
    AssertThrow(false, ExcMessage("Newton iteration failed to converge"));
    return max_newton_iterations;
  }

  Vector<float> Problem::recovery_indicators() const
  {
    Vector<float> indicators(triangulation.n_active_cells());
    std::vector<Tensor<1, 2>> recovered(dof_handler.n_dofs());
    std::vector<unsigned int> counts(dof_handler.n_dofs(), 0);
    QTrapezoid<2> vertex_quadrature;
    FEValues<2> vertex_values(fe, vertex_quadrature, update_gradients);
    std::vector<Tensor<1, 2>> gradients_at_vertices(vertex_quadrature.size());
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        vertex_values.reinit(cell);
        vertex_values.get_function_gradients(solution, gradients_at_vertices);
        cell->get_dof_indices(local_dofs);
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
          {
            const unsigned int local = fe.component_to_system_index(0, v);
            recovered[local_dofs[local]] += gradients_at_vertices[v];
            ++counts[local_dofs[local]];
          }
      }
    for (unsigned int i = 0; i < recovered.size(); ++i)
      if (counts[i] > 0)
        recovered[i] /= static_cast<double>(counts[i]);

    const QGauss<2> quadrature(fe.degree + 2);
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
              recovered_q += fe_values.shape_value(i, q) *
                             recovered[local_dofs[i]];
            eta_squared += (recovered_q - gradients[q]).norm_square() *
                           fe_values.JxW(q);
          }
        indicators[cell->active_cell_index()] = std::sqrt(eta_squared);
      }
    return indicators;
  }

  void Problem::mark_and_adapt(const Vector<float> &indicators)
  {
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                       indicators,
                                                       top_fraction,
                                                       bottom_fraction);
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell : triangulation.active_cell_iterators_on_level(0))
      cell->clear_coarsen_flag();
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
    // The paper reports more than 2,000 iterations for the initial fit and
    // fewer than 100 for warm-started later fits.
    const unsigned int iterations = surrogate.is_trained() ? 100 : 3000;
    const auto result = surrogate.train(points, values, iterations);
    training_file << std::fixed << std::setprecision(8) << time << ','
                  << dof_handler.n_dofs() << ',' << result.closure_evaluations
                  << ',' << std::scientific << std::setprecision(12)
                  << result.final_loss << ',' << std::fixed
                  << std::setprecision(6) << result.seconds << '\n';
    training_file.flush();
    std::cout << "  NN samples=" << dof_handler.n_dofs()
              << " evaluations=" << result.closure_evaluations
              << " loss=" << std::scientific << result.final_loss
              << " seconds=" << std::fixed << result.seconds << std::endl;
    return result;
  }

  void Problem::update_nn_previous_vector()
  {
    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(),
                                         dof_handler,
                                         support_points);
    std::vector<std::array<double, 2>> points(dof_handler.n_dofs());
    for (const auto &entry : support_points)
      points[entry.first] = {entry.second[0], entry.second[1]};
    const auto values = surrogate.values(points);
    for (unsigned int i = 0; i < values.size(); ++i)
      nn_previous[i] = values[i];
  }

  double Problem::compute_mass() const
  {
    const QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe, quadrature, update_values | update_JxW_values);
    std::vector<double> values(quadrature.size());
    double mass = 0.0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, values);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          mass += values[q] * fe_values.JxW(q);
      }
    return mass;
  }

  double Problem::compute_energy() const
  {
    const QGauss<2> quadrature(fe.degree + 2);
    FEValues<2> fe_values(fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);
    std::vector<double> values(quadrature.size());
    std::vector<Tensor<1, 2>> gradients(quadrature.size());
    double energy = 0.0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, values);
        fe_values.get_function_gradients(solution, gradients);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const double potential = 0.25 * std::pow(1.0 - values[q] * values[q], 2) /
                                     (epsilon * epsilon);
            energy += (0.5 * gradients[q].norm_square() + potential) *
                      fe_values.JxW(q);
          }
      }
    return energy;
  }

  std::pair<double, double> Problem::solution_range() const
  {
    return {*std::min_element(solution.begin(), solution.end()),
            *std::max_element(solution.begin(), solution.end())};
  }

  void Problem::output_results(const unsigned int output_index,
                               const double time,
                               const Vector<float> &indicators)
  {
    update_nn_previous_vector();
    Vector<float> levels(triangulation.n_active_cells());
    for (const auto &cell : triangulation.active_cell_iterators())
      levels[cell->active_cell_index()] = cell->level();

    DataOut<2> out;
    out.attach_dof_handler(dof_handler);
    out.add_data_vector(solution, "solution");
    out.add_data_vector(nn_previous, "nn_previous_solution");
    out.add_data_vector(indicators, "error_indicator");
    out.add_data_vector(levels, "refinement_level");
    out.build_patches();

    const std::string basename =
      "allen-cahn-" + Utilities::int_to_string(output_index, 4) + ".vtu";
    std::ofstream file(output_directory + "/" + basename);
    out.write_vtu(file);
    pvd_records.emplace_back(time, basename);
    write_pvd();
  }

  void Problem::write_pvd() const
  {
    std::ofstream pvd(output_directory + "/allen-cahn.pvd");
    DataOutBase::write_pvd_record(pvd, pvd_records);
  }

  void Problem::run()
  {
    std::filesystem::create_directories(output_directory);
    history_file.open(output_directory + "/simulation-history.csv");
    newton_file.open(output_directory + "/newton-history.csv");
    training_file.open(output_directory + "/neural-training.csv");
    AssertThrow(history_file && newton_file && training_file,
                ExcMessage("Could not create output CSV files"));
    history_file << "step,time,adaptive_iterations,active_cells,dofs,newton_iterations,"
                    "estimator,mass,energy,min_solution,max_solution,elapsed_seconds\n";
    newton_file << "step,time,iteration,residual,relative_residual\n";
    training_file << "time,samples,closure_evaluations,final_mse,seconds\n";
    run_start = std::chrono::steady_clock::now();

    make_initial_mesh();
    setup_system();
    Vector<float> indicators;
    // Recovery-driven initial adaptations. The paper-aligned default is five.
    for (unsigned int cycle = 0; cycle < initial_adaptations; ++cycle)
      {
        set_initial_condition();
        indicators = recovery_indicators();
        std::cout << "Initial adapt " << cycle + 1
                  << " cells=" << triangulation.n_active_cells()
                  << " dofs=" << dof_handler.n_dofs()
                  << " eta=" << indicators.l2_norm() << std::endl;
        if (cycle + 1 < initial_adaptations)
          mark_and_adapt(indicators);
      }

    train_surrogate(0.0);
    update_nn_previous_vector();
    output_results(0, 0.0, indicators);
    const auto initial_range = solution_range();
    history_file << 0 << ',' << 0.0 << ',' << initial_adaptations << ','
                 << triangulation.n_active_cells() << ',' << dof_handler.n_dofs()
                 << ',' << 0 << ',' << indicators.l2_norm() << ','
                 << compute_mass() << ',' << compute_energy() << ','
                 << initial_range.first << ',' << initial_range.second << ','
                 << 0.0 << '\n';

    const unsigned int n_steps =
      static_cast<unsigned int>(std::llround(final_time / time_step));
    for (unsigned int step = 1; step <= n_steps; ++step)
      {
        const double time = step * time_step;
        unsigned int used_adaptations = 0;
        unsigned int newton_iterations = 0;
        std::cout << "Time step " << step << '/' << n_steps
                  << " t=" << std::fixed << std::setprecision(4) << time
                  << std::endl;
        for (unsigned int adapt = 0; adapt < max_adaptive_iterations; ++adapt)
          {
            initialize_newton_guess_from_surrogate();
            newton_iterations = solve_newton(step, time);
            indicators = recovery_indicators();
            ++used_adaptations;

            Vector<float> h1_cells(triangulation.n_active_cells());
            Vector<double> zero(dof_handler.n_dofs());
            VectorTools::integrate_difference(dof_handler,
                                              solution,
                                              Functions::ZeroFunction<2>(),
                                              h1_cells,
                                              QGauss<2>(fe.degree + 2),
                                              VectorTools::H1_seminorm);
            const double h1 = VectorTools::compute_global_error(
              triangulation, h1_cells, VectorTools::H1_seminorm);
            const double relative_estimator = indicators.l2_norm() /
                                              std::max(h1, 1e-14);
            std::cout << "  Adapt " << adapt + 1
                      << " cells=" << triangulation.n_active_cells()
                      << " dofs=" << dof_handler.n_dofs()
                      << " relative_eta=" << relative_estimator << std::endl;
            if (relative_estimator <= estimator_tolerance ||
                adapt + 1 == max_adaptive_iterations)
              break;
            mark_and_adapt(indicators);
          }

        const auto range = solution_range();
        const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start)
            .count();
        history_file << step << ',' << std::setprecision(12) << time << ','
                     << used_adaptations << ','
                     << triangulation.n_active_cells() << ','
                     << dof_handler.n_dofs() << ',' << newton_iterations << ','
                     << indicators.l2_norm() << ',' << compute_mass() << ','
                     << compute_energy() << ',' << range.first << ','
                     << range.second << ',' << elapsed << '\n';
        history_file.flush();
        output_results(step, time, indicators);
        train_surrogate(time);
      }
  }
} // namespace AllenCahn

int main(const int argc, char **argv)
{
  try
    {
      const double final_time =
        argc > 1 ? std::stod(argv[1]) : AllenCahn::default_final_time;
      const std::string output_directory =
        argc > 2 ? argv[2] : "runs/example-4p7-nn";
      const unsigned int max_adapt =
        argc > 3 ? static_cast<unsigned int>(std::stoul(argv[3])) : 7;
      const double estimator_tolerance =
        argc > 4 ? std::stod(argv[4]) : 0.5;
      const unsigned int initial_subdivisions =
        argc > 5 ? static_cast<unsigned int>(std::stoul(argv[5])) : 8;
      const unsigned int max_grid_level =
        argc > 6 ? static_cast<unsigned int>(std::stoul(argv[6])) : 8;
      const unsigned int initial_adaptations =
        argc > 7 ? static_cast<unsigned int>(std::stoul(argv[7])) : 5;
      const double time_step =
        argc > 8 ? std::stod(argv[8]) : AllenCahn::default_time_step;
      AssertThrow(final_time > 0.0,
                  ExcMessage("final_time must be positive"));
      AssertThrow(time_step > 0.0,
                  ExcMessage("time_step must be positive"));
      AssertThrow(std::abs(final_time / time_step -
                           std::round(final_time / time_step)) < 1e-10,
                  ExcMessage("final_time must be a multiple of time_step"));
      AssertThrow(max_adapt > 0,
                  ExcMessage("max_adapt_iterations must be positive"));
      AssertThrow(estimator_tolerance > 0.0,
                  ExcMessage("estimator_tolerance must be positive"));
      AssertThrow(initial_subdivisions > 0,
                  ExcMessage("initial_subdivisions must be positive"));
      AssertThrow(initial_adaptations > 0,
                  ExcMessage("initial_adaptations must be positive"));

      deallog.depth_console(0);
      AllenCahn::Problem problem(final_time,
                                 output_directory,
                                 max_adapt,
                                 estimator_tolerance,
                                 initial_subdivisions,
                                 max_grid_level,
                                 initial_adaptations,
                                 time_step);
      problem.run();
    }
  catch (const std::exception &exception)
    {
      std::cerr << "\nERROR: " << exception.what() << std::endl;
      return 1;
    }
  return 0;
}

