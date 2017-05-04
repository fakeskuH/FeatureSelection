// DimRedux.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <algorithm>
#include <armadillo>

using namespace std;
using namespace arma;

void pca(mat *data, mat *projection, vec *eigenvals) {
	mat coeff;

	princomp(coeff, *projection, *eigenvals, *data);

	data->print();
	cout << endl;
	coeff.print();
	cout << endl;
	eigenvals->print();
	cout << endl;
	projection->print();
	cout << endl;
}

double calculateVariance(vec feature) {
	double variance = 0;
	if (feature.n_elem > 1) {

		double mean = accu(feature) / feature.n_elem;
		vec submean(feature);

		submean.for_each([mean](arma::vec::elem_type& val) { val -= mean; val = pow(val, 2); });
		variance = accu(submean) / (submean.n_elem - 1);
	}
	return variance;
}

void removeDuplicateFeatures(mat *data, unsigned int id, double margin) {
	for (unsigned int i = 0; i < data->n_cols; ++i) {
		if (i == id) continue;
		vec comp = data->col(i);
		vec original = data->col(id);
		for (unsigned int j = 0; j < original.n_elem; ++j) {
			if (original.at(j) < (comp.at(j) - margin) || original.at(j) > (comp.at(j) + margin))
				data->shed_col(i);
		}
	}
}

void forwardFeatureSelection(mat *data, mat *selectedFeatures, double targetAccuracy, bool usePCA=true, double varianceThreshold = 0) {
	double crossValidationAccuracy = 0;

	if (usePCA) {
		mat projection;
		vec eigenvals;

		pca(data, &projection, &eigenvals);

		unsigned int i = 0;
		while (crossValidationAccuracy < targetAccuracy) {
			if (i >= projection.n_cols) {
				cout << "All features are added. Unable to reach target accuracy." << endl;
				break;
			}
			int index_max = eigenvals.index_max();
			selectedFeatures->insert_cols(selectedFeatures->n_cols, projection.col(index_max));
			eigenvals.shed_row(index_max);

			//crossValidate(&selectedFeatures, &crossValidationAccuracy);
			crossValidationAccuracy += 10;
			cout << "Cross validation accuracy: " << crossValidationAccuracy << endl;
			++i;
		}
		if (crossValidationAccuracy >= targetAccuracy) cout << "Target accuracy reached." << endl;
	} else { // No PCA

		// Prune features
		for (unsigned int i = 0; i < data->n_cols; ++i) {
			// Step 1) Remove low variance features
			if (calculateVariance(data->col(i)) < varianceThreshold) data->shed_col(i);
			// Step 2) Remove duplicate features
			removeDuplicateFeatures(data, i, 0.1);
		}
		
		// Start selection
		while (crossValidationAccuracy < targetAccuracy) {
			double maxAccuracy = 0;
			unsigned int maxAccuracyId = 0;

			for (unsigned int i = 0; i < data->n_cols; ++i) {
				selectedFeatures->insert_cols(selectedFeatures->n_cols, data->col(i));
				//crossValidate(&selectedFeatures, &crossValidationAccuracy);
				if (crossValidationAccuracy > maxAccuracy) {
					maxAccuracy = crossValidationAccuracy;
					maxAccuracyId = i;
				} else selectedFeatures->shed_col(selectedFeatures->n_cols - 1);
			}

			data->shed_col(maxAccuracyId);
		}
	}
}

int main()
{

	mat A, selectedFeatures;
	mat projection;
	vec eigenvals;

	A
		<< -1 << 1 << 0 << endr
		<< -2 << 2 << 0 << endr
		<< 2 << -2 << 0 << endr
		<< 1 << -1 << 0 << endr
		<< -1 << -1 << 0 << endr
		<< -3 << -3 << 0 << endr
		<< 3 << 3 << 0 << endr
		<< 1 << 1 << 0 << endr;

	forwardFeatureSelection(&A, &selectedFeatures, 30);
	getchar();
    return 0;
}

/* 
	1) Remove low variance features
	2) Remove duplicate features
	3) 
*/