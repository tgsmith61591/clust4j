/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.metrics.pairwise;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.AbstractCentroidClusterer;
import com.clust4j.algo.CentroidClustererParameters;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.utils.VecUtils;

public class HaversineTest {
	final static double[][] coordinates = new double[][] {
		new double[]{30.2500, 97.7500}, // Austin, TX
		new double[]{32.7767, 96.7970}, // Dallas, TX
		new double[]{29.7604, 95.3698}, // Houston, TX
		new double[]{40.7903, 73.9597}, // Manhattan
		new double[]{40.7484, 73.9857}  // Empire State Bldg
	};

	@Test
	public void test1() {
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(coordinates, false);
		StandardScaler scaler = new StandardScaler().fit(mat);
		
		AbstractCentroidClusterer km;
		CentroidClustererParameters<? extends AbstractCentroidClusterer> planner;
		
		planner = new KMeansParameters(2)
				.setVerbose(true)
				.setMetric(Distance.HAVERSINE.MI)
				.setVerbose(true);
		km = planner.fitNewModel(scaler.transform(mat));
		
		int[] kmlabels = km.getLabels();
		assertTrue(kmlabels[0] == kmlabels[1] && kmlabels[1] == kmlabels[2]);
		assertTrue(kmlabels[1] != kmlabels[3] && kmlabels[3] == kmlabels[4]);
	}
	
	@Test
	public void test2() {

		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(coordinates, false);
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		
		AbstractCentroidClusterer km;
		CentroidClustererParameters<? extends AbstractCentroidClusterer> planner;
		
		planner = new KMedoidsParameters(2)
			.setVerbose(true)
			.setMetric(Distance.HAVERSINE.MI)
			.setVerbose(true);
		km = planner.fitNewModel(X);
		
		int[] kmlabels = km.getLabels();
		assertTrue(kmlabels[0] == kmlabels[1] && kmlabels[1] == kmlabels[2]);
		assertTrue(kmlabels[1] != kmlabels[3] && kmlabels[3] == kmlabels[4]);
		assertTrue(kmlabels[0] == 0 && kmlabels[1] == 0 && kmlabels[2] == 0);
		

		// Inverse transform the centroid:
		double[][] c = new double[][]{km.getCentroids().get(0)};
		Array2DRowRealMatrix cm = new Array2DRowRealMatrix(c, false);
		RealMatrix inverse = scaler.inverseTransform(cm);
		
		assertTrue( VecUtils.equalsExactly(inverse.getRow(0), coordinates[0]) ); // First one should be Austin
	}

}
