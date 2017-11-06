/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
// @HEADER
// ***********************************************************************
//
//       xSDKTrilinos: Extreme-scale Software Development Kit Package
//                 Copyright (2016) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alicia Klinvex    (amklinv@sandia.gov)
//                    James Willenbring (jmwille@sandia.gov)
//                    Michael Heroux    (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

// Several pieces of this codebase has been adapted from the HYPRE interface of
// xSDKTrilinos package.

#include "HypreDirectSolver.h"

namespace sierra {
namespace nalu {

namespace {
// This anonymous namespace contains wrapper methods to HYPRE solver creation
// methods. It hides around the fact that some solvers require an MPI
// communicator while others do not. This allows HypreDirectSolver::CreateSolver
// methods to assign pointers using the same function signature.
//
// Note this section has been adapted from the xSDK Trilinos package

int Hypre_BoomerAMGCreate(MPI_Comm, HYPRE_Solver* solver)
{ return HYPRE_BoomerAMGCreate(solver); }

int Hypre_ParaSailsCreate(MPI_Comm comm, HYPRE_Solver* solver)
{ return HYPRE_ParaSailsCreate(comm, solver); }

int Hypre_EuclidCreate(MPI_Comm comm, HYPRE_Solver* solver)
{ return HYPRE_EuclidCreate(comm, solver); }

int Hypre_AMSCreate(MPI_Comm, HYPRE_Solver *solver)
{ return HYPRE_AMSCreate(solver);}

int Hypre_ParCSRHybridCreate(MPI_Comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRHybridCreate(solver);}

int Hypre_ParCSRPCGCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRPCGCreate(comm, solver);}

int Hypre_ParCSRGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRGMRESCreate(comm, solver);}

int Hypre_ParCSRFlexGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRFlexGMRESCreate(comm, solver);}

int Hypre_ParCSRLGMRESCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRLGMRESCreate(comm, solver);}

int Hypre_ParCSRBiCGSTABCreate(MPI_Comm comm, HYPRE_Solver *solver)
{ return HYPRE_ParCSRBiCGSTABCreate(comm, solver);}
}

HypreDirectSolver::HypreDirectSolver(
  std::string name,
  HypreLinearSolverConfig* config,
  LinearSolvers* linearSolvers
) : LinearSolver(name, linearSolvers, config)
{}

HypreDirectSolver::~HypreDirectSolver()
{
  destroyLinearSolver();
}

int
HypreDirectSolver::solve(
  int& numIterations,
  double& finalResidualNorm)
{
  // Initialize the solver on first entry
  if (!isInitialized_) initSolver();

  numIterations = 0;
  finalResidualNorm = 0.0;

  // Solve the system Ax = b
  solverSolvePtr_(solver_, parMat_, parRhs_, parSln_);

  // Extract linear num. iterations and linear residual. Unlike the TPetra
  // interface, Hypre returns the relative residual norm and not the final
  // absolute residual.
  solverNumItersPtr_(solver_, &numIterations);
  solverFinalResidualNormPtr_(solver_, &finalResidualNorm);

  return 0;
}

void
HypreDirectSolver::destroyLinearSolver()
{
  if (isSolverSetup_) solverDestroyPtr_(solver_);
  isSolverSetup_ = false;

  if (isPrecondSetup_) precondDestroyPtr_(precond_);
  isPrecondSetup_ = false;
}

void
HypreDirectSolver::initSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  auto plist = config_->paramsPrecond();

  solverType_ = plist->get("Solver", Hypre::GMRES);
  usePrecond_ = plist->get("SetPreconditioner", false);
  if (usePrecond_)
    precondType_ = plist->get("Preconditioner", Hypre::BoomerAMG);

  Hypre::Hypre_Chooser chooser = plist->get("SolveOrPrecondition", Hypre::Solver);
  if (chooser != Hypre::Solver)
    throw std::runtime_error(
      "HypreDirectSolver::initParameters: Invalid option provided for Hypre Solver");

  // Everything checks out... create the solver and preconditioner
  createSolver();
  if (usePrecond_) createPrecond();

  // Apply user configuration parameters to solver and precondtioner
  int numFuncs = plist->get("NumFunctions", 0);
  if (numFuncs > 0) {
    Teuchos::RCP<Ifpack2::FunctionParameter>* params =
      plist->get<Teuchos::RCP<Ifpack2::FunctionParameter>*>("Functions");

    for (int i=0; i < numFuncs; i++) {
      params[i]->CallFunction(solver_, precond_);
    }
  }

  if (usePrecond_)
    solverPrecondPtr_(solver_, precondSolvePtr_, precondSetupPtr_, precond_);

  // We are always using HYPRE solver
  solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);

  isInitialized_ = true;
}

void
HypreDirectSolver::createSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isSolverSetup_) {
    solverDestroyPtr_(solver_);
    isSolverSetup_ = false;
  }

  switch(solverType_) {
  case Hypre::BoomerAMG:
    solverCreatePtr_ = &Hypre_BoomerAMGCreate;
    solverDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
    solverSetupPtr_ = &HYPRE_BoomerAMGSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &HYPRE_BoomerAMGSolve;
    solverNumItersPtr_ = &HYPRE_BoomerAMGGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
    break;

  case Hypre::GMRES:
    solverCreatePtr_ = &Hypre_ParCSRGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRGMRESSolve;
    solverNumItersPtr_ = &HYPRE_GMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_GMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::FlexGMRES:
    solverCreatePtr_ = &Hypre_ParCSRFlexGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRFlexGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRFlexGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRFlexGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRFlexGMRESSolve;
    solverNumItersPtr_ = &HYPRE_FlexGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_FlexGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::LGMRES:
    solverCreatePtr_ = &Hypre_ParCSRLGMRESCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRLGMRESDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRLGMRESSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRLGMRESSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRLGMRESSolve;
    solverNumItersPtr_ = &HYPRE_LGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_LGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::BiCGSTAB:
    solverCreatePtr_ = &Hypre_ParCSRBiCGSTABCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRBiCGSTABDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRBiCGSTABSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRBiCGSTABSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRBiCGSTABSolve;
    solverNumItersPtr_ = &HYPRE_BiCGSTABGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_BiCGSTABGetFinalRelativeResidualNorm;
    break;

  case Hypre::AMS:
    solverCreatePtr_ = &Hypre_AMSCreate;
    solverDestroyPtr_ = &HYPRE_AMSDestroy;
    solverSetupPtr_ = &HYPRE_AMSSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &HYPRE_AMSSolve;
    solverNumItersPtr_ = &HYPRE_AMSGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_AMSGetFinalRelativeResidualNorm;
    break;

  case Hypre::PCG:
    solverCreatePtr_ = &Hypre_ParCSRPCGCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRPCGDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRPCGSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRPCGSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRPCGSolve;
    solverNumItersPtr_ = &HYPRE_PCGGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_PCGGetFinalRelativeResidualNorm;
    break;

  case Hypre::Hybrid:
    solverCreatePtr_ = &Hypre_ParCSRHybridCreate;
    solverDestroyPtr_ = &HYPRE_ParCSRHybridDestroy;
    solverSetupPtr_ = &HYPRE_ParCSRHybridSetup;
    solverPrecondPtr_ = &HYPRE_ParCSRHybridSetPrecond;
    solverSolvePtr_ = &HYPRE_ParCSRHybridSolve;
    solverNumItersPtr_ = &HYPRE_ParCSRHybridGetNumIterations;
    solverFinalResidualNormPtr_ = &HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
    break;

  default:
    solverCreatePtr_ = nullptr;
    break;
  }

  if (solverCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing HYPRE Solver");

  solverCreatePtr_(comm_, &solver_);
  isSolverSetup_ = true;
}

void
HypreDirectSolver::createPrecond()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isPrecondSetup_) {
    precondDestroyPtr_(solver_);
    isPrecondSetup_ = false;
  }

  switch(precondType_) {
  case Hypre::BoomerAMG:
    precondCreatePtr_ = &Hypre_BoomerAMGCreate;
    precondDestroyPtr_ = &HYPRE_BoomerAMGDestroy;
    precondSetupPtr_ = &HYPRE_BoomerAMGSetup;
    precondSolvePtr_ = &HYPRE_BoomerAMGSolve;
    break;

  case Hypre::Euclid:
    precondCreatePtr_ = &Hypre_EuclidCreate;
    precondDestroyPtr_ = &HYPRE_EuclidDestroy;
    precondSetupPtr_ = &HYPRE_EuclidSetup;
    precondSolvePtr_ = &HYPRE_EuclidSolve;
    break;

  case Hypre::ParaSails:
    precondCreatePtr_ = &Hypre_ParaSailsCreate;
    precondDestroyPtr_ = &HYPRE_ParaSailsDestroy;
    precondSetupPtr_ = &HYPRE_ParaSailsSetup;
    precondSolvePtr_ = &HYPRE_ParaSailsSolve;
    break;

  case Hypre::AMS:
    precondCreatePtr_ = &Hypre_AMSCreate;
    precondDestroyPtr_ = &HYPRE_AMSDestroy;
    precondSetupPtr_ = &HYPRE_AMSSetup;
    precondSolvePtr_ = &HYPRE_AMSSolve;
    break;

  default:
    precondCreatePtr_ = nullptr;
    break;
  }

  if (precondCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing HYPRE Preconditioner");

  precondCreatePtr_(comm_, &precond_);
  isPrecondSetup_ = true;
}

}  // nalu
}  // sierra
