/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef HYPREDIRECTSOLVER_H
#define HYPREDIRECTSOLVER_H

#include "LinearSolver.h"

#include "Ifpack2_Hypre.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

namespace sierra {
namespace nalu {

/** Hypre Direct Solver
 *
 *  This solver bypassess the Trilinos/Tpetra interface and manages the STK to
 *  Hypre interface directly.
 */
class HypreDirectSolver: public LinearSolver
{
public:
  HypreDirectSolver(
    std::string name,
    HypreLinearSolverConfig* config,
    LinearSolvers* linearSolvers);

  virtual ~HypreDirectSolver();

  /** Unused pure virtual method required to be overridden
   *
   *  Since we don't use Tpetra data structures, this method is unused
   */
  virtual void setupLinearSolver(
    Teuchos::RCP<LinSys::Vector>,
    Teuchos::RCP<LinSys::Matrix>,
    Teuchos::RCP<LinSys::Vector>,
    Teuchos::RCP<LinSys::MultiVector>) override
  {}

  virtual void destroyLinearSolver() override;

  virtual  int solve(
    Teuchos::RCP<LinSys::Vector>,
    int & ,
    double & ) override
  {
    throw std::runtime_error("Bad call to HypreDirectSolver::solve");
    return 0;
  }

  /** Solves the linear system and updates the solution vector.
   *
   *  @param iters The number of linear iterations performed
   *  @param norm The norm of the final relative residual
   */
  int solve(int&, double&);

  virtual PetraType getType() override { return PT_HYPRE; }

  mutable HYPRE_ParCSRMatrix parMat_;

  mutable HYPRE_ParVector parRhs_;

  mutable HYPRE_ParVector parSln_;

  MPI_Comm comm_;

private:
  void initSolver();

  void createSolver();

  void createPrecond();

  Ifpack2::Hypre::Hypre_Solver solverType_;
  Ifpack2::Hypre::Hypre_Solver precondType_;

  mutable HYPRE_Solver solver_;

  mutable HYPRE_Solver precond_;

  int (*solverCreatePtr_)(MPI_Comm, HYPRE_Solver*);
  int (*solverDestroyPtr_)(HYPRE_Solver);
  int (*solverSetupPtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  int (*solverSolvePtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  int (*solverPrecondPtr_)(
    HYPRE_Solver,
    HYPRE_PtrToParSolverFcn,
    HYPRE_PtrToParSolverFcn,
    HYPRE_Solver);

  int (*precondCreatePtr_)(MPI_Comm, HYPRE_Solver*);
  int (*precondDestroyPtr_)(HYPRE_Solver);
  int (*precondSetupPtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  int (*precondSolvePtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);

  int (*solverNumItersPtr_)(HYPRE_Solver, int*);
  int (*solverFinalResidualNormPtr_)(HYPRE_Solver, double*);


  bool usePrecond_{false};
  bool isSolverSetup_{false};
  bool isPrecondSetup_{false};
  bool isInitialized_{false};
};

}  // nalu
}  // sierra

#endif /* HYPREDIRECTSOLVER_H */
