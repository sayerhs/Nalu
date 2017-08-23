/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef LinearSolverConfig_h
#define LinearSolverConfig_h

#include <string>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace Ifpack2 {
class FunctionParameter;
}

namespace YAML {
class Node;
}

namespace sierra{
namespace nalu{

class LinearSolverConfig
{
public:
  LinearSolverConfig();
  virtual ~LinearSolverConfig() = default;

  virtual void load(const YAML::Node&) = 0;

  inline std::string name() const
  { return name_ ; }

  const Teuchos::RCP<Teuchos::ParameterList> & params() const
  { return params_; }

  const Teuchos::RCP<Teuchos::ParameterList> & paramsPrecond() const
  { return paramsPrecond_; }

  inline bool getWriteMatrixFiles() const
  { return writeMatrixFiles_; }

  inline bool recomputePreconditioner() const
  { return recomputePreconditioner_; }

  inline bool reusePreconditioner() const
  { return reusePreconditioner_; }

  std::string get_method() const
  {return method_;}

  std::string preconditioner_type() const
  { return preconditionerType_;}

protected:
  std::string name_;
  std::string method_;
  std::string precond_;
  std::string preconditionerType_{"RELAXATION"};

  Teuchos::RCP<Teuchos::ParameterList> params_;
  Teuchos::RCP<Teuchos::ParameterList> paramsPrecond_;

  bool recomputePreconditioner_{false};
  bool reusePreconditioner_{false};
  bool writeMatrixFiles_{false};
};

class TpetraLinearSolverConfig : public LinearSolverConfig
{
public:
  TpetraLinearSolverConfig();
  virtual ~TpetraLinearSolverConfig();

  virtual void load(const YAML::Node & node) final;
  bool getSummarizeMueluTimer() { return summarizeMueluTimer_; }
  std::string & muelu_xml_file() {return muelu_xml_file_;}
  bool use_MueLu() const {return useMueLu_;}

private:
  std::string muelu_xml_file_;
  bool summarizeMueluTimer_{false};
  bool useMueLu_{false};
};

class HypreLinearSolverConfig : public LinearSolverConfig
{
public:
  HypreLinearSolverConfig();

  virtual ~HypreLinearSolverConfig() {};

  virtual void load(const YAML::Node&);

protected:
  std::vector<Teuchos::RCP<Ifpack2::FunctionParameter>> funcParams_;
  double tolerance_{1.0e-4};
  int maxIterations_{50};
  int outputLevel_{1};
  int kspace_{1};

  /* BoomerAMG options */
  double bamgStrongThreshold_{0.57};
  int bamgCoarsenType_{6};
  int bamgCycleType_{1};
  int bamgRelaxType_{6};
  int bamgRelaxOrder_{1};
  int bamgNumSweeps_{2};
  int bamgMaxLevels_{20};
  int bamgInterpType_{0};
  std::string bamgEuclidFile_{""};

  bool isHypreSolver_{true};

private:
  void boomerAMG_solver_config(const YAML::Node&);
  void boomerAMG_precond_config(const YAML::Node&);

  void euclid_precond_config(const YAML::Node&);

  void hypre_gmres_solver_config(const YAML::Node&);
  void hypre_pcg_solver_config(const YAML::Node&);

  void configure_hypre_preconditioner(const YAML::Node&);
  void configure_hypre_solver(const YAML::Node&);

};


} // namespace nalu
} // namespace Sierra

#endif
