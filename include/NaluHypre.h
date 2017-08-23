/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NALUHYPRE_H
#define NALUHYPRE_H

#include "Ifpack2_Hypre.hpp"
#include "Realm.h"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/MetaData.hpp"

#include <cmath>

namespace Ifpack2 {

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class NaluHypre:
    public Ifpack2_Hypre<Scalar, LocalOrdinal, GlobalOrdinal, Node>
{
public:
  NaluHypre(
    const Teuchos::RCP<const Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
    sierra::nalu::Realm* realm);

  virtual ~NaluHypre() {}

  virtual void initialize();

  virtual void apply(const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >& X,
                       Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >& Y,
                       Teuchos::ETransp mode = Teuchos::NO_TRANS,
                       Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                       Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

  inline void dumpMatrixFiles(const bool dowrite)
  { writeMatrixFiles_ = dowrite; }

protected:
  sierra::nalu::Realm* realm_;

  int outputCounter_{0};

  bool writeMatrixFiles_{false};
};

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
NaluHypre<Scalar,LocalOrdinal,GlobalOrdinal,Node>::NaluHypre(
  const Teuchos::RCP<const Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >& A,
  sierra::nalu::Realm* realm
): Ifpack2::Ifpack2_Hypre<Scalar, LocalOrdinal, GlobalOrdinal, Node>(A),
  realm_(realm)
{
  this->hypreILower_ = realm_->hypreILower_;
  this->hypreIUpper_ = realm_->hypreIUpper_;
  this->initHypreDataStructures();
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void NaluHypre<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initialize(){
  using Teuchos::RCP;

  // Create a timer
  const std::string timerName ("Ifpack2::Hypre::initialize");
  RCP<Teuchos::Time> timer = Teuchos::TimeMonitor::lookupCounter (timerName);
  if (timer.is_null ()) {
    timer = Teuchos::TimeMonitor::getNewCounter (timerName);
  }

  { // Start timer here
    Teuchos::TimeMonitor timeMon (*timer);

    MPI_Comm comm = this->GetMpiComm();
    auto& bulk = realm_->bulk_data();

    int nprocs, iproc;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &iproc);

    HYPRE_IJMatrixCreate(comm,
                         this->hypreILower_,
                         this->hypreIUpper_,
                         this->hypreILower_,
                         this->hypreIUpper_, &(this->HypreA_));
    HYPRE_IJMatrixSetObjectType(this->HypreA_, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(this->HypreA_);

    std::vector<bool> rowFilled(this->hypreIUpper_-this->hypreILower_+1, false);

    for(size_t i = 0; i < this->A_->getNodeNumRows(); i++){
      int numElements = this->A_->getNumEntriesInLocalRow(i);
      Teuchos::Array<LocalOrdinal> indices(numElements);
      Teuchos::Array<Scalar> values(numElements);
      std::vector<int> idlist(0);
      std::vector<double> valuelist(0);
      size_t numEntries;
      this->A_->getLocalRowCopy(i, indices(), values(), numEntries);
      int GlobalRow[1];
      GlobalRow[0] = this->A_->getRowMap()->getGlobalElement(i) - 1;
      auto rnode = bulk.get_entity(
        stk::topology::NODE_RANK, GlobalRow[0]+1);
      int *rowid = stk::mesh::field_data(*realm_->hypreGlobalId_, rnode);
      GlobalRow[0] = rowid[0];
      for(size_t j = 0; j < numEntries; j++){
        int gidx = this->A_->getColMap()->getGlobalElement(indices[j]);
          auto node = bulk.get_entity(
            stk::topology::NODE_RANK, gidx);
          if (bulk.is_valid(node)) {
            int* hid = stk::mesh::field_data(*realm_->hypreGlobalId_, node);
            idlist.push_back(hid[0]);
            valuelist.push_back(values[j]);
          } else {
            if (std::fabs(values[j]) > 1.0e-8)
              std::cerr << this->Comm()->getRank() << "\t"
                        << this->A_->getRowMap()->getGlobalElement(i)
                        << "\t" << gidx << "\t" << values[j] << std::endl;
          }
      }
      int nelems = idlist.size();
      HYPRE_IJMatrixAddToValues(this->HypreA_, 1, &nelems, GlobalRow, &idlist[0], &valuelist[0]);
      rowFilled[GlobalRow[0]-this->hypreILower_] = true;
    }

    // Populate rows that were inactive in the Tpetra Matrix
    int hnrows=1;
    int hncols=1;
    double getval;
    double setval=1.0;
    for (int i=this->hypreILower_; i<=this->hypreIUpper_; i++) {
      if (rowFilled[i-this->hypreILower_]) continue;
      HYPRE_IJMatrixGetValues(this->HypreA_, hnrows, &hncols, &i, &i, &getval);
      if (std::fabs(getval) < 1.0e-8) {
        HYPRE_IJMatrixAddToValues(this->HypreA_, hnrows, &hncols, &i, &i, &setval);
      }
    }
    HYPRE_IJMatrixAssemble(this->HypreA_);
    HYPRE_IJMatrixGetObject(this->HypreA_, (void**)&this->ParMatrix_);
  } // Stop timer here

  if (writeMatrixFiles_)
    HYPRE_IJMatrixPrint(this->HypreA_, "IJMatrix.A");

  this->isInitialized_=true;
  this->numInitialize_++;
  this->initializeTime_ = timer->totalElapsedTime();
} //initialize()

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void NaluHypre<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply(const Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >& X,
                         Tpetra::MultiVector< Scalar, LocalOrdinal, GlobalOrdinal, Node >& Y,
                         Teuchos::ETransp mode, // TODO: These parameters currently do nothing
                         Scalar alpha,
                         Scalar beta) const
{
  // Create a timer
  const std::string timerName ("Ifpack2::Hypre::apply");
  Teuchos::RCP<Teuchos::Time> timer = Teuchos::TimeMonitor::lookupCounter (timerName);
  if (timer.is_null ()) {
    timer = Teuchos::TimeMonitor::getNewCounter (timerName);
  }

  TEUCHOS_TEST_FOR_EXCEPTION(!this->isComputed(), std::runtime_error,
         Teuchos::typeName (*this) << "::apply(): Preconditioner has not been computed.");

  TEUCHOS_TEST_FOR_EXCEPTION(!this->A_->getDomainMap()->isSameAs(*X.getMap()), std::runtime_error,
      Teuchos::typeName (*this) << "::apply(): X's map must match A's domain map.");

  TEUCHOS_TEST_FOR_EXCEPTION(!this->A_->getRangeMap()->isSameAs(*Y.getMap()), std::runtime_error,
      Teuchos::typeName (*this) << "::apply(): Y's map must match A's range map.");

  auto& bulk = realm_->bulk_data();
  { // Start timer here
    Teuchos::TimeMonitor timeMon (*timer);

    size_t NumVectors = X.getNumVectors();
    TEUCHOS_TEST_FOR_EXCEPTION(NumVectors != Y.getNumVectors(), std::runtime_error,
         Teuchos::typeName (*this) << "::apply(): X and Y must have the same number of vectors.");

    HYPRE_ParVectorSetConstantValues(this->ParX_, 0.0);
    for (size_t VecNum=0; VecNum < NumVectors; VecNum++) {
        auto rhsVec = X.getVector(VecNum);
        auto rhsView = rhsVec->get1dView();
        auto rhsMap = rhsVec->getMap();
        auto limin = rhsMap->getMinLocalIndex();
        auto limax = rhsMap->getMaxLocalIndex();

        for (int i=limin; i <= limax; i++) {
            int ig = rhsMap->getGlobalElement(i);
            auto node = bulk.get_entity(
              stk::topology::NODE_RANK, ig);
            int* ih = stk::mesh::field_data(*realm_->hypreGlobalId_, node);
            double value = rhsView[i];
            HYPRE_IJVectorAddToValues(this->XHypre_, 1, ih, &value);
        }
        HYPRE_IJVectorAssemble(this->XHypre_);
        HYPRE_IJVectorGetObject(this->XHypre_, (void**) &this->ParX_);
        // HYPRE_IJMatrixPrint(HypreA_, "IJ.out.A");
        // HYPRE_IJVectorPrint(this->XHypre_, "IJ.out.b");

        HYPRE_ParVectorSetConstantValues(this->ParY_, 0.0);
        if(this->SolveOrPrec_ == Hypre::Solver){
            // Use the solver methods
            this->SolverSolvePtr_(this->Solver_, this->ParMatrix_,
                            this->ParX_, this->ParY_);
        } else {
            // Apply the preconditioner
            this->PrecondSolvePtr_(
              this->Preconditioner_, this->ParMatrix_,
              this->ParX_, this->ParY_);
        }

        auto slnData = Y.getDataNonConst(VecNum);
        double outval;
        for (int i=limin; i <= limax; i++) {
            int ig = rhsMap->getGlobalElement(i);
            auto node = bulk.get_entity(
              stk::topology::NODE_RANK, ig);
            int* ih = stk::mesh::field_data(*realm_->hypreGlobalId_, node);
            HYPRE_IJVectorGetValues(this->YHypre_, 1, ih, &outval);
            slnData[i] = outval;
        }
        // HYPRE_IJVectorPrint(this->YHypre_,"IJ.out.x");
        this->SolverNumItersPtr_(this->Solver_, &this->hypreNumLinIters_);
        this->SolverFinalResidualNormPtr_(this->Solver_, &this->hypreFinalResidual_);
    }
  } // Stop timer here

  if (writeMatrixFiles_) {
    HYPRE_IJVectorPrint(this->XHypre_, "IJVector.b");
    HYPRE_IJVectorPrint(this->YHypre_, "IJVector.x");
  }

  this->numApply_++;
  this->applyTime_ = timer->totalElapsedTime();
} //ApplyInverse()
}  // nalu


#endif /* NALUHYPRE_H */
