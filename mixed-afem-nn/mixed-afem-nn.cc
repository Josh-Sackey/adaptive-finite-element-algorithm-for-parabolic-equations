#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

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
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "neural_surrogate.h"

using namespace dealii;

namespace MixedAFEM
{
  constexpr double pi = numbers::PI;

  enum class Example { sine = 1, gaussian = 2 };
  enum class Transfer { classical, neural };

  class ManufacturedSolution : public Function<2>
  {
  public:
    ManufacturedSolution(const Example example, const double time)
      : Function<2>(2, time), example(example)
    {}

    double value(const Point<2> &p, const unsigned int component = 0) const override
    {
      const double t = this->get_time();
      if (example == Example::sine)
        {
          const double phi = std::exp(-t) * std::sin(2*pi*p[0]) * std::sin(2*pi*p[1]);
          return component == 0 ? phi : 8*pi*pi*phi;
        }
      const double c = 0.3 * std::cos(2*pi*t);
      const double dx = p[0] - c, dy = p[1] - c;
      const double s = dx*dx + dy*dy;
      const double phi = std::exp(-500.0*s);
      return component == 0 ? phi : (2000.0 - 1000000.0*s)*phi;
    }

    Tensor<1,2> gradient(const Point<2> &p,
                         const unsigned int component = 0) const override
    {
      Tensor<1,2> g;
      const double t = this->get_time();
      if (example == Example::sine)
        {
          g[0] = std::exp(-t)*2*pi*std::cos(2*pi*p[0])*std::sin(2*pi*p[1]);
          g[1] = std::exp(-t)*2*pi*std::sin(2*pi*p[0])*std::cos(2*pi*p[1]);
          if (component == 1) g *= 8*pi*pi;
          return g;
        }
      const double beta = 500.0;
      const double c = 0.3 * std::cos(2*pi*t);
      const double dx = p[0]-c, dy = p[1]-c;
      const double s = dx*dx + dy*dy;
      const double phi = std::exp(-beta*s);
      if (component == 0)
        {
          g[0] = -2*beta*dx*phi;
          g[1] = -2*beta*dy*phi;
        }
      else
        {
          const double factor = (8*beta*beta*beta*s - 16*beta*beta)*phi;
          g[0] = factor*dx;
          g[1] = factor*dy;
        }
      return g;
    }

    double forcing(const Point<2> &p) const
    {
      const double t = this->get_time();
      if (example == Example::sine)
        return 64*std::pow(pi,4) * value(p,0);
      const double beta = 500.0;
      const double c = 0.3*std::cos(2*pi*t);
      const double cp = -0.6*pi*std::sin(2*pi*t);
      const double dx = p[0]-c, dy = p[1]-c;
      const double s = dx*dx + dy*dy;
      const double phi = std::exp(-beta*s);
      const double phi_t = 2*beta*(dx*cp + dy*cp)*phi;
      const double bilaplacian =
        (16*std::pow(beta,4)*s*s - 64*std::pow(beta,3)*s + 32*beta*beta)*phi;
      return phi_t + bilaplacian + phi;
    }

  private:
    const Example example;
  };

  class Problem
  {
  public:
    Problem(Example, Transfer, double final_time, double time_step,
            const std::string &output_directory);
    void run();

  private:
    void make_grid();
    void setup_system();
    void set_exact_initial(double time);
    void assemble(double time);
    void solve();
    Vector<float> recovery_indicators() const;
    void adapt_exact(double time);
    TrainingResult train_surrogate(double time);
    void adapt_for_transfer(double time);
    std::pair<double,double> errors(double time) const;
    void output(unsigned int index, double time, const Vector<float> &indicator);
    void record(const std::string &phase, unsigned int index, double time,
                double solve_seconds, double transfer_seconds,
                const TrainingResult &training = {});

    const Example example;
    const Transfer transfer;
    const double final_time;
    const double dt;
    const std::filesystem::path output_dir;
    const double top_fraction = 0.6;
    const double bottom_fraction = 0.4;
    const unsigned int max_level = 5;

    Triangulation<2> triangulation;
    FESystem<2> fe;
    DoFHandler<2> dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern sparsity;
    SparseMatrix<double> matrix;
    Vector<double> solution, old_solution, rhs;
    NeuralSurrogate surrogate;
    std::ofstream history, training_history;
    std::vector<std::pair<double,std::string>> pvd_records;
    const FEValuesExtractors::Scalar phi{0}, mu{1};
  };

  Problem::Problem(const Example e, const Transfer tr, const double T,
                   const double time_step, const std::string &directory)
    : example(e), transfer(tr), final_time(T), dt(time_step),
      output_dir(directory), fe(FE_Q<2>(1),2), dof_handler(triangulation)
  {}

  void Problem::make_grid()
  {
    const Point<2> lower = example == Example::sine ? Point<2>(0,0) : Point<2>(-1,-1);
    const Point<2> upper = Point<2>(1,1);
    GridGenerator::subdivided_hyper_rectangle(triangulation, {8,8}, lower, upper);
  }

  void Problem::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,dsp,constraints,false);
    sparsity.copy_from(dsp);
    matrix.reinit(sparsity);
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }

  void Problem::set_exact_initial(const double time)
  {
    ManufacturedSolution exact(example,time);
    VectorTools::interpolate(dof_handler,exact,solution);
    constraints.distribute(solution);
    old_solution = solution;
  }

  void Problem::assemble(const double time)
  {
    matrix = 0; rhs = 0;
    QGauss<2> quad(fe.degree+2);
    QGauss<1> face_quad(fe.degree+2);
    FEValues<2> values(fe,quad,update_values|update_gradients|
                      update_quadrature_points|update_JxW_values);
    FEFaceValues<2> face_values(fe,face_quad,update_values|update_quadrature_points|
                               update_normal_vectors|update_JxW_values);
    FullMatrix<double> cell_matrix(fe.dofs_per_cell,fe.dofs_per_cell);
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    std::vector<double> old_phi(quad.size());
    ManufacturedSolution exact(example,time);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix=0; cell_rhs=0; values.reinit(cell);
        values[phi].get_function_values(old_solution,old_phi);
        for (unsigned int q=0;q<quad.size();++q)
          for (unsigned int i=0;i<fe.dofs_per_cell;++i)
            {
              const double vi=values[phi].value(i,q);
              const double wi=values[mu].value(i,q);
              cell_rhs(i) += vi*(old_phi[q]/dt + exact.forcing(values.quadrature_point(q))) * values.JxW(q);
              for (unsigned int j=0;j<fe.dofs_per_cell;++j)
                cell_matrix(i,j) +=
                  ((1.0/dt+1.0)*vi*values[phi].value(j,q)
                   + values[phi].gradient(i,q)*values[mu].gradient(j,q)
                   + values[mu].gradient(i,q)*values[phi].gradient(j,q)
                   - wi*values[mu].value(j,q))*values.JxW(q);
            }
        for (const auto face_no : cell->face_indices())
          if (cell->face(face_no)->at_boundary())
            {
              face_values.reinit(cell,face_no);
              for (unsigned int q=0;q<face_quad.size();++q)
                {
                  const auto normal=face_values.normal_vector(q);
                  const double g1=exact.gradient(face_values.quadrature_point(q),0)*normal;
                  const double g2=exact.gradient(face_values.quadrature_point(q),1)*normal;
                  for (unsigned int i=0;i<fe.dofs_per_cell;++i)
                    cell_rhs(i) += (g2*face_values[phi].value(i,q)
                                    +g1*face_values[mu].value(i,q))*face_values.JxW(q);
                }
            }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cell_matrix,cell_rhs,local,matrix,rhs);
      }
  }

  void Problem::solve()
  {
    SparseDirectUMFPACK direct;
    direct.initialize(matrix);
    direct.vmult(solution,rhs);
    constraints.distribute(solution);
  }

  Vector<float> Problem::recovery_indicators() const
  {
    Vector<float> indicators(triangulation.n_active_cells());
    std::vector<Tensor<1,2>> recovered(dof_handler.n_dofs());
    std::vector<unsigned int> counts(dof_handler.n_dofs(),0);
    QTrapezoid<2> vertices;
    FEValues<2> vertex_values(fe,vertices,update_gradients);
    std::vector<Tensor<1,2>> gradients(vertices.size());
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell:dof_handler.active_cell_iterators())
      {
        vertex_values.reinit(cell);
        vertex_values[phi].get_function_gradients(solution,gradients);
        cell->get_dof_indices(local);
        for (unsigned int v=0;v<GeometryInfo<2>::vertices_per_cell;++v)
          {
            const unsigned int li=fe.component_to_system_index(0,v);
            recovered[local[li]] += gradients[v];
            ++counts[local[li]];
          }
      }
    for (unsigned int i=0;i<recovered.size();++i)
      if (counts[i]) recovered[i]/=static_cast<double>(counts[i]);

    QGauss<2> quad(fe.degree+2);
    FEValues<2> values(fe,quad,update_values|update_gradients|update_JxW_values);
    std::vector<Tensor<1,2>> numerical(quad.size());
    for (const auto &cell:dof_handler.active_cell_iterators())
      {
        values.reinit(cell);
        values[phi].get_function_gradients(solution,numerical);
        cell->get_dof_indices(local);
        double eta2=0;
        for (unsigned int q=0;q<quad.size();++q)
          {
            Tensor<1,2> recovered_q;
            for (unsigned int i=0;i<fe.dofs_per_cell;++i)
              recovered_q += values[phi].value(i,q)*recovered[local[i]];
            eta2 += (recovered_q-numerical[q]).norm_square()*values.JxW(q);
          }
        indicators[cell->active_cell_index()]=std::sqrt(eta2);
      }
    return indicators;
  }

  void Problem::adapt_exact(const double time)
  {
    auto indicator=recovery_indicators();
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,indicator,
                                                       top_fraction,bottom_fraction);
    if (triangulation.n_levels()>max_level)
      for (const auto &cell:triangulation.active_cell_iterators_on_level(max_level))
        cell->clear_refine_flag();
    triangulation.execute_coarsening_and_refinement();
    setup_system();
    set_exact_initial(time);
  }

  TrainingResult Problem::train_surrogate(const double time)
  {
    std::map<types::global_dof_index,Point<2>> support;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(),dof_handler,support);
    const IndexSet phi_dofs=DoFTools::extract_dofs(dof_handler,fe.component_mask(phi));
    std::vector<std::array<double,2>> points;
    std::vector<double> data;
    points.reserve(phi_dofs.n_elements()); data.reserve(phi_dofs.n_elements());
    for (auto i=phi_dofs.begin();i!=phi_dofs.end();++i)
      {
        const auto &p=support[*i];
        points.push_back({p[0],p[1]}); data.push_back(solution[*i]);
      }
    const unsigned int iterations=surrogate.is_trained()?100:500;
    const auto result=surrogate.train(points,data,iterations);
    training_history << std::fixed << std::setprecision(8) << time << ','
                     << points.size() << ',' << result.closure_evaluations << ','
                     << std::scientific << result.final_loss << ',' << std::fixed
                     << result.seconds << '\n';
    return result;
  }

  void Problem::adapt_for_transfer(const double time)
  {
    const auto indicator=recovery_indicators();
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,indicator,
                                                       top_fraction,bottom_fraction);
    if (triangulation.n_levels()>max_level)
      for (const auto &cell:triangulation.active_cell_iterators_on_level(max_level))
        cell->clear_refine_flag();

    if (transfer==Transfer::classical)
      {
        Vector<double> previous=solution;
        SolutionTransfer<2,Vector<double>> st(dof_handler);
        triangulation.prepare_coarsening_and_refinement();
        st.prepare_for_coarsening_and_refinement(previous);
        triangulation.execute_coarsening_and_refinement();
        setup_system();
        st.interpolate(old_solution);
        constraints.distribute(old_solution);
      }
    else
      {
        triangulation.execute_coarsening_and_refinement();
        setup_system();
        std::map<types::global_dof_index,Point<2>> support;
        DoFTools::map_dofs_to_support_points(MappingQ1<2>(),dof_handler,support);
        const IndexSet phi_dofs=DoFTools::extract_dofs(dof_handler,fe.component_mask(phi));
        std::vector<std::array<double,2>> points;
        std::vector<types::global_dof_index> indices;
        for (auto i=phi_dofs.begin();i!=phi_dofs.end();++i)
          { const auto &p=support[*i]; points.push_back({p[0],p[1]}); indices.push_back(*i); }
        const auto predicted=surrogate.values(points);
        old_solution=0;
        for (unsigned int k=0;k<indices.size();++k) old_solution[indices[k]]=predicted[k];
        constraints.distribute(old_solution);
      }
    solution=old_solution;
    (void)time;
  }

  std::pair<double,double> Problem::errors(const double time) const
  {
    ManufacturedSolution exact(example,time);
    Vector<float> difference(triangulation.n_active_cells());
    ComponentSelectFunction<2> phi_weight(0,2), mu_weight(1,2);
    VectorTools::integrate_difference(dof_handler,solution,exact,difference,
      QGauss<2>(fe.degree+3),VectorTools::L2_norm,&phi_weight);
    const double ephi=VectorTools::compute_global_error(triangulation,difference,
                                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,solution,exact,difference,
      QGauss<2>(fe.degree+3),VectorTools::L2_norm,&mu_weight);
    const double emu=VectorTools::compute_global_error(triangulation,difference,
                                                       VectorTools::L2_norm);
    return {ephi,emu};
  }

  void Problem::record(const std::string &phase,const unsigned int index,
                       const double time,const double solve_seconds,
                       const double transfer_seconds,const TrainingResult &tr)
  {
    const auto [ephi,emu]=errors(time);
    history << phase << ',' << index << ',' << std::fixed << std::setprecision(8)
            << time << ',' << triangulation.n_active_cells() << ','
            << dof_handler.n_dofs() << ',' << std::scientific << ephi << ','
            << emu << ',' << std::fixed << solve_seconds << ',' << transfer_seconds
            << ',' << tr.seconds << ',' << std::scientific << tr.final_loss << '\n';
    history.flush();
    std::cout << phase << " t=" << std::fixed << std::setprecision(2) << time
              << " cells=" << triangulation.n_active_cells() << " dofs="
              << dof_handler.n_dofs() << std::scientific << " e_phi=" << ephi
              << " e_mu=" << emu << std::fixed << " solve_s=" << solve_seconds
              << " transfer_s=" << transfer_seconds << std::endl;
  }

  void Problem::output(const unsigned int index,const double time,
                       const Vector<float> &indicator)
  {
    Vector<double> exact_values(dof_handler.n_dofs()), error_values(dof_handler.n_dofs());
    ManufacturedSolution exact(example,time);
    VectorTools::interpolate(dof_handler,exact,exact_values);
    error_values=solution; error_values-=exact_values;
    Vector<float> levels(triangulation.n_active_cells());
    for (const auto &cell:triangulation.active_cell_iterators())
      levels[cell->active_cell_index()]=cell->level();
    const std::vector<std::string> names={"phi","mu"};
    DataOut<2> out; out.attach_dof_handler(dof_handler);
    out.add_data_vector(solution,names);
    const std::vector<std::string> exact_names={"exact_phi","exact_mu"};
    const std::vector<std::string> error_names={"error_phi","error_mu"};
    out.add_data_vector(exact_values,exact_names);
    out.add_data_vector(error_values,error_names);
    out.add_data_vector(indicator,"error_indicator");
    out.add_data_vector(levels,"refinement_level");
    out.build_patches();
    const std::string filename="solution-"+Utilities::int_to_string(index,4)+".vtu";
    std::ofstream file(output_dir/filename); out.write_vtu(file);
    pvd_records.emplace_back(time,filename);
    std::ofstream pvd(output_dir/"solution.pvd");
    DataOutBase::write_pvd_record(pvd,pvd_records);
  }

  void Problem::run()
  {
    std::filesystem::create_directories(output_dir);
    history.open(output_dir/"results.csv");
    training_history.open(output_dir/"neural-training.csv");
    history << "phase,index,time,active_cells,dofs,l2_error_phi,l2_error_mu,solve_seconds,transfer_seconds,training_seconds,training_mse\n";
    training_history << "time,samples,closure_evaluations,normalized_mse,seconds\n";
    make_grid(); setup_system(); set_exact_initial(0.0);
    for (unsigned int cycle=0;cycle<5;++cycle)
      {
        record("initial",cycle,0.0,0,0);
        output(cycle,0.0,recovery_indicators());
        if (cycle<4) adapt_exact(0.0);
      }

    const unsigned int n_steps=static_cast<unsigned int>(std::lround(final_time/dt));
    for (unsigned int step=1;step<=n_steps;++step)
      {
        const double time=step*dt;
        const auto solve_start=std::chrono::steady_clock::now();
        assemble(time); solve();
        const double solve_seconds=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-solve_start).count();
        auto indicator=recovery_indicators();
        output(4+step,time,indicator);
        record("time_step",step,time,solve_seconds,0.0);
        TrainingResult tr;
        double transfer_seconds=0;
        if (step<n_steps)
          {
            const auto transfer_start=std::chrono::steady_clock::now();
            if (transfer==Transfer::neural) tr=train_surrogate(time);
            adapt_for_transfer(time);
            transfer_seconds=std::chrono::duration<double>(
              std::chrono::steady_clock::now()-transfer_start).count();
            record("transfer_check",step,time,0.0,transfer_seconds,tr);
          }
      }
  }
}

int main(int argc,char **argv)
{
  try
    {
      AssertThrow(argc==6,ExcMessage(
        "Usage: mixed-afem-nn <1|2> <classical|nn> <final_time> <dt> <output_dir>"));
      const auto example=std::stoi(argv[1])==1?MixedAFEM::Example::sine:MixedAFEM::Example::gaussian;
      const std::string method=argv[2];
      AssertThrow(method=="classical"||method=="nn",ExcMessage("method must be classical or nn"));
      MixedAFEM::Problem problem(example,
        method=="classical"?MixedAFEM::Transfer::classical:MixedAFEM::Transfer::neural,
        std::stod(argv[3]),std::stod(argv[4]),argv[5]);
      problem.run();
    }
  catch (const std::exception &e)
    { std::cerr << "ERROR: " << e.what() << std::endl; return 1; }
  return 0;
}
