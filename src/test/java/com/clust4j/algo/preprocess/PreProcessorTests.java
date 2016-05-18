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
package com.clust4j.algo.preprocess;

import static org.junit.Assert.*;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class PreProcessorTests {

	@Test
	public void testMeanCenter() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};

		final Array2DRowRealMatrix d = new Array2DRowRealMatrix(data, false);
		MeanCenterer norm = new MeanCenterer().fit(d);
		RealMatrix scaled = norm.transform(d);
		double[][] operated = scaled.getData();
		
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 0)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 1)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 2)), 0, Precision.EPSILON));
		
		// test copy
		operated = norm.copy().transform(data);
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 0)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 1)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 2)), 0, Precision.EPSILON));
		
		// Test inverse transform.
		assertTrue(MatUtils.equalsWithTolerance(data, norm.inverseTransform(scaled).getData(), 1e-8));
		
		// copy coverage
		norm.copy();
		
		// test dim mismatch
		boolean a = false;
		try {
			norm.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// test not fit
		a = false;
		try {
			new MeanCenterer().transform(d);
		} catch(ModelNotFitException dim) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testBoxCoxTransformer() {
		Array2DRowRealMatrix iris = TestSuite.IRIS_DATASET.getData();
		BoxCoxTransformer bc = new BoxCoxTransformer().fit(iris);
		
		RealMatrix X = bc.transform(iris);
		
		// make sure it works...
		bc.inverseTransform(X);
		
		// we suffer some accuracy issues on inverse transform due to log bases, etc...
		//assertTrue(MatUtils.equalsWithTolerance(bc.inverseTransform(X).getData(), iris.getData(), 1.0));
		
		// Test a large matrix...
		Array2DRowRealMatrix big = TestSuite.getRandom(400, 5);
		bc = new BoxCoxTransformer().fit(big);
		X = bc.transform(big);
		
		// test dim mismatch
		boolean a = false;
		try {
			bc.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// test not fit
		a = false;
		try {
			new BoxCoxTransformer().transform(iris);
		} catch(ModelNotFitException dim) { a = true; }
		finally { assertTrue(a); }
		
		// Test too small:
		a = false;
		try {
			new BoxCoxTransformer().fit(TestSuite.getRandom(1, 5));
		} catch(IllegalArgumentException i) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testYJTransformer() {
		Array2DRowRealMatrix iris = TestSuite.IRIS_DATASET.getData();
		YeoJohnsonTransformer bc = new YeoJohnsonTransformer().fit(iris);
		
		RealMatrix X = bc.transform(iris);
		//System.out.println(TestSuite.formatter.format(X));
		//System.out.println(Arrays.toString(bc.lambdas));
		
		// make sure it works...
		bc.inverseTransform(X);
		//System.out.println(TestSuite.formatter.format(bc.inverseTransform(X)));
		
		// we suffer some accuracy issues on inverse transform due to log bases, etc...
		//assertTrue(MatUtils.equalsWithTolerance(bc.inverseTransform(X).getData(), iris.getData(), 1.0));
		
		// Test a large matrix...
		Array2DRowRealMatrix big = TestSuite.getRandom(400, 5);
		bc = new YeoJohnsonTransformer().fit(big);
		X = bc.transform(big);
		
		// test dim mismatch
		boolean a = false;
		try {
			bc.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// test not fit
		a = false;
		try {
			new YeoJohnsonTransformer().transform(iris);
		} catch(ModelNotFitException dim) { a = true; }
		finally { assertTrue(a); }
		
		// Test too small:
		a = false;
		try {
			new YeoJohnsonTransformer().fit(TestSuite.getRandom(1, 5));
		} catch(IllegalArgumentException i) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testMedianCenter() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};

		final Array2DRowRealMatrix d = new Array2DRowRealMatrix(data, false);
		MedianCenterer norm = new MedianCenterer().fit(d);
		RealMatrix scaled = norm.transform(d);
		double[][] operated = scaled.getData();
		
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 0)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 1)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 2)), 0, Precision.EPSILON));
		
		// test copy
		operated = norm.copy().transform(data);
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 0)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 1)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.median(MatUtils.getColumn(operated, 2)), 0, Precision.EPSILON));
		
		// Test inverse transform.
		assertTrue(MatUtils.equalsWithTolerance(data, norm.inverseTransform(scaled).getData(), 1e-8));
		
		// copy coverage
		norm.copy();
		
		// test dim mismatch
		boolean a = false;
		try {
			norm.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// test not fit
		a = false;
		try {
			new MedianCenterer().transform(d);
		} catch(ModelNotFitException dim) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testRobustScaler() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};

		final Array2DRowRealMatrix d = new Array2DRowRealMatrix(data, false);
		RobustScaler norm = new RobustScaler().fit(d);
		RealMatrix scaled = norm.transform(d);
		
		// Test inverse transform.
		assertTrue(MatUtils.equalsWithTolerance(data, norm.inverseTransform(scaled).getData(), 1e-8));
		
		// copy coverage
		norm.copy();
		
		// test dim mismatch
		boolean a = false;
		try {
			norm.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// test not fit
		a = false;
		try {
			new RobustScaler().transform(d);
		} catch(ModelNotFitException dim) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testCenterScale() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix d = new Array2DRowRealMatrix(data, false);
		final StandardScaler norm = new StandardScaler().fit(d);
		final RealMatrix normed = norm.transform(d);
		final double[][] operated = normed.getData();
		
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 0)), 0, 1e-12));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 1)), 0, 1e-12));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 2)), 0, 1e-12));
		
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 0)), 1, 1e-12));
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 1)), 1, 1e-12));
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 2)), 1, 1e-12));
		
		// test inverse transform
		assertTrue(MatUtils.equalsWithTolerance(data, norm.inverseTransform(normed).getData(), 1e-8));
		
		// test dim mismatch
		boolean a = false;
		try {
			norm.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
		
		// assert that fewer than two features will throw exception
		a = false;
		try {
			norm.fit(TestSuite.getRandom(1, 4));
		} catch(IllegalArgumentException i) {
			a = true;
		} finally { assertTrue(a); }
	}
	
	@Test
	public void testMinMaxScale() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};

		final Array2DRowRealMatrix d = new Array2DRowRealMatrix(data, false);
		final MinMaxScaler norm = new MinMaxScaler().fit(d);
		final RealMatrix normed = norm.transform(d);
		final double[][] operated = normed.getData();
		
		for(int i = 0; i < operated[0].length; i++) {
			double[] col = MatUtils.getColumn(operated, i);
			assertTrue(VecUtils.min(col) >= 0);
			assertTrue(VecUtils.max(col) <= 1);
		}
		
		// test inverse transform
		assertTrue(MatUtils.equalsWithTolerance(data, norm.inverseTransform(normed).getData(), 1e-8));
		
		// test dim mismatch
		boolean a = false;
		try {
			norm.inverseTransform(TestSuite.getRandom(2, 2));
		} catch(DimensionMismatchException dim) { a = true; }
		finally { assertTrue(a); }
	}
	
	@Test
	public void testMinMaxScalerBadMinMax() {
		boolean a = false;
		
		try {
			double[][] d = new double[][]{
				new double[]{1,2,3},
				new double[]{1,2,3}
			};
			
			final Array2DRowRealMatrix data = new Array2DRowRealMatrix(d, false);
			new MinMaxScaler(1, 0).fit(data);
		} catch(IllegalStateException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testScalersNotFit() {
		boolean a = false;
		try {
			new StandardScaler().transform(TestSuite.IRIS_DATASET.getData());
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new StandardScaler().fit(TestSuite.IRIS_DATASET.getData())
				.transform(new double[][]{new double[]{1}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new PCA(1).transform(TestSuite.IRIS_DATASET.getData());
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new PCA(1).fit(TestSuite.IRIS_DATASET.getData())
				.transform(new double[][]{new double[]{1}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new MinMaxScaler().transform(TestSuite.IRIS_DATASET.getData());
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new MinMaxScaler().fit(TestSuite.IRIS_DATASET.getData())
				.transform(new double[][]{new double[]{1}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new MeanCenterer().transform(TestSuite.IRIS_DATASET.getData());
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new MeanCenterer().fit(TestSuite.IRIS_DATASET.getData())
				.transform(new double[][]{new double[]{1}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testPCA() {
		final Array2DRowRealMatrix X = TestSuite.IRIS_DATASET.getData();
		PCA pca = new PCA(2).fit(X);
		RealMatrix xp = pca.transform(X);
		
		double[][] expected = new double[][]{
			new double[]{-2.68420713, -0.32660731},
			new double[]{-2.71539062,  0.16955685},
			new double[]{-2.88981954,  0.13734561},
			new double[]{-2.7464372 ,  0.31112432},
			new double[]{-2.72859298, -0.33392456},
			new double[]{-2.27989736, -0.74778271},
			new double[]{-2.82089068,  0.08210451},
			new double[]{-2.62648199, -0.17040535},
			new double[]{-2.88795857,  0.57079803},
			new double[]{-2.67384469,  0.1066917 },
			new double[]{-2.50652679, -0.65193501},
			new double[]{-2.61314272, -0.02152063},
			new double[]{-2.78743398,  0.22774019},
			new double[]{-3.22520045,  0.50327991},
			new double[]{-2.64354322, -1.1861949 },
			new double[]{-2.38386932, -1.34475434},
			new double[]{-2.6225262 , -0.81808967},
			new double[]{-2.64832273, -0.31913667},
			new double[]{-2.19907796, -0.87924409},
			new double[]{-2.58734619, -0.52047364},
			new double[]{-2.3105317 , -0.39786782},
			new double[]{-2.54323491, -0.44003175},
			new double[]{-3.21585769, -0.14161557},
			new double[]{-2.30312854, -0.10552268},
			new double[]{-2.35617109,  0.03120959},
			new double[]{-2.50791723,  0.13905634},
			new double[]{-2.469056  , -0.13788731},
			new double[]{-2.56239095, -0.37468456},
			new double[]{-2.63982127, -0.31929007},
			new double[]{-2.63284791,  0.19007583},
			new double[]{-2.58846205,  0.19739308},
			new double[]{-2.41007734, -0.41808001},
			new double[]{-2.64763667, -0.81998263},
			new double[]{-2.59715948, -1.10002193},
			new double[]{-2.67384469,  0.1066917 },
			new double[]{-2.86699985, -0.0771931 },
			new double[]{-2.62522846, -0.60680001},
			new double[]{-2.67384469,  0.1066917 },
			new double[]{-2.98184266,  0.48025005},
			new double[]{-2.59032303, -0.23605934},
			new double[]{-2.77013891, -0.27105942},
			new double[]{-2.85221108,  0.93286537},
			new double[]{-2.99829644,  0.33430757},
			new double[]{-2.4055141 , -0.19591726},
			new double[]{-2.20883295, -0.44269603},
			new double[]{-2.71566519,  0.24268148},
			new double[]{-2.53757337, -0.51036755},
			new double[]{-2.8403213 ,  0.22057634},
			new double[]{-2.54268576, -0.58628103},
			new double[]{-2.70391231, -0.11501085},
			new double[]{ 1.28479459, -0.68543919},
			new double[]{ 0.93241075, -0.31919809},
			new double[]{ 1.46406132, -0.50418983},
			new double[]{ 0.18096721,  0.82560394},
			new double[]{ 1.08713449, -0.07539039},
			new double[]{ 0.64043675,  0.41732348},
			new double[]{ 1.09522371, -0.28389121},
			new double[]{-0.75146714,  1.00110751},
			new double[]{ 1.04329778, -0.22895691},
			new double[]{-0.01019007,  0.72057487},
			new double[]{-0.5110862 ,  1.26249195},
			new double[]{ 0.51109806,  0.10228411},
			new double[]{ 0.26233576,  0.5478933 },
			new double[]{ 0.98404455,  0.12436042},
			new double[]{-0.174864  ,  0.25181557},
			new double[]{ 0.92757294, -0.46823621},
			new double[]{ 0.65959279,  0.35197629},
			new double[]{ 0.23454059,  0.33192183},
			new double[]{ 0.94236171,  0.54182226},
			new double[]{ 0.0432464 ,  0.58148945},
			new double[]{ 1.11624072,  0.08421401},
			new double[]{ 0.35678657,  0.06682383},
			new double[]{ 1.29646885,  0.32756152},
			new double[]{ 0.92050265,  0.18239036},
			new double[]{ 0.71400821, -0.15037915},
			new double[]{ 0.89964086, -0.32961098},
			new double[]{ 1.33104142, -0.24466952},
			new double[]{ 1.55739627, -0.26739258},
			new double[]{ 0.81245555,  0.16233157},
			new double[]{-0.30733476,  0.36508661},
			new double[]{-0.07034289,  0.70253793},
			new double[]{-0.19188449,  0.67749054},
			new double[]{ 0.13499495,  0.31170964},
			new double[]{ 1.37873698,  0.42120514},
			new double[]{ 0.58727485,  0.48328427},
			new double[]{ 0.8072055 , -0.19505396},
			new double[]{ 1.22042897, -0.40803534},
			new double[]{ 0.81286779,  0.370679  },
			new double[]{ 0.24519516,  0.26672804},
			new double[]{ 0.16451343,  0.67966147},
			new double[]{ 0.46303099,  0.66952655},
			new double[]{ 0.89016045,  0.03381244},
			new double[]{ 0.22887905,  0.40225762},
			new double[]{-0.70708128,  1.00842476},
			new double[]{ 0.35553304,  0.50321849},
			new double[]{ 0.33112695,  0.21118014},
			new double[]{ 0.37523823,  0.29162202},
			new double[]{ 0.64169028, -0.01907118},
			new double[]{-0.90846333,  0.75156873},
			new double[]{ 0.29780791,  0.34701652},
			new double[]{ 2.53172698,  0.01184224},
			new double[]{ 1.41407223,  0.57492506},
			new double[]{ 2.61648461, -0.34193529},
			new double[]{ 1.97081495,  0.18112569},
			new double[]{ 2.34975798,  0.04188255},
			new double[]{ 3.39687992, -0.54716805},
			new double[]{ 0.51938325,  1.19135169},
			new double[]{ 2.9320051 , -0.35237701},
			new double[]{ 2.31967279,  0.24554817},
			new double[]{ 2.91813423, -0.78038063},
			new double[]{ 1.66193495, -0.2420384 },
			new double[]{ 1.80234045,  0.21615461},
			new double[]{ 2.16537886, -0.21528028},
			new double[]{ 1.34459422,  0.77641543},
			new double[]{ 1.5852673 ,  0.53930705},
			new double[]{ 1.90474358, -0.11881899},
			new double[]{ 1.94924878, -0.04073026},
			new double[]{ 3.48876538, -1.17154454},
			new double[]{ 3.79468686, -0.25326557},
			new double[]{ 1.29832982,  0.76101394},
			new double[]{ 2.42816726, -0.37678197},
			new double[]{ 1.19809737,  0.60557896},
			new double[]{ 3.49926548, -0.45677347},
			new double[]{ 1.38766825,  0.20403099},
			new double[]{ 2.27585365, -0.33338653},
			new double[]{ 2.61419383, -0.55836695},
			new double[]{ 1.25762518,  0.179137  },
			new double[]{ 1.29066965,  0.11642525},
			new double[]{ 2.12285398,  0.21085488},
			new double[]{ 2.3875644 , -0.46251925},
			new double[]{ 2.84096093, -0.37274259},
			new double[]{ 3.2323429 , -1.37052404},
			new double[]{ 2.15873837,  0.21832553},
			new double[]{ 1.4431026 ,  0.14380129},
			new double[]{ 1.77964011,  0.50146479},
			new double[]{ 3.07652162, -0.68576444},
			new double[]{ 2.14498686, -0.13890661},
			new double[]{ 1.90486293, -0.04804751},
			new double[]{ 1.16885347,  0.1645025 },
			new double[]{ 2.10765373, -0.37148225},
			new double[]{ 2.31430339, -0.18260885},
			new double[]{ 1.92245088, -0.40927118},
			new double[]{ 1.41407223,  0.57492506},
			new double[]{ 2.56332271, -0.2759745 },
			new double[]{ 2.41939122, -0.30350394},
			new double[]{ 1.94401705, -0.18741522},
			new double[]{ 1.52566363,  0.37502085},
			new double[]{ 1.76404594, -0.07851919},
			new double[]{ 1.90162908, -0.11587675},
			new double[]{ 1.38966613,  0.28288671}	
		};
		
		/*
		 * We have to test everything at the abs level, because
		 * commons math produces some sign swappage in SVD...
		 */
		assertTrue(VecUtils.equalsWithTolerance(pca.getVariabilityExplained(), new double[]{4.19667516,  0.24062861}, 1e-4));
		assertTrue(VecUtils.equalsWithTolerance(pca.getVariabilityRatioExplained(), new double[]{0.92461621,  0.05301557}, 1e-4));
		
		
		assertTrue(MatUtils.equalsWithTolerance(MatUtils.abs(pca.getComponents().getData()), new double[][]{
			new double[]{ 0.36158968,  0.08226889,  0.85657211,  0.35884393},
			new double[]{ 0.65653988,  0.72971237,  0.1757674 ,  0.07470647}
		}, 1e-3));
		
		assertTrue(MatUtils.equalsWithTolerance(MatUtils.abs(expected), MatUtils.abs(xp.getData()), 1e-4));
		
		
		// Test inverse transform... we definitely get some bad floating precision here...
		RealMatrix inverse = pca.inverseTransform(xp);
		assertTrue(MatUtils.equalsWithTolerance(TestSuite.IRIS_DATASET.getData().getData(), inverse.getData(), 0.75));
		
		
		
		/*
		 * Test exceptions...
		 */
		boolean a = false;
		try {
			pca.transform(new double[][]{new double[]{1.0,1.0,1.0}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new PCA(0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new PCA(0.0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			new PCA(1.1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		// Test copy:
		new PCA(1).copy(); // test works on non-fit
		PCA copy = pca.copy().fit(X); // refit on the same data
		
		// assert fit, basically:
		assertTrue(MatUtils.equalsExactly(xp.getData(), copy.transform(X).getData()));
		assertTrue(copy.getNoiseVariance() == pca.getNoiseVariance());
		assertTrue(VecUtils.equalsExactly(pca.getCumulativeVariabilityRatioExplained(), 
			new double[]{
				0.9246162071742684, 
				0.9776317750248033
		}));
		
		PCA p = new PCA(15).fit(TestSuite.IRIS_DATASET.getData());
		assertTrue(p.getComponents().getColumnDimension() == 4);
	}
	
	@Test
	public void testWeightTransformer() {
		RealMatrix iris = TestSuite.IRIS_DATASET.getData();
		
		// first test on 1.0 weights, assert same.
		double[] weights = VecUtils.rep(1.0, 4);
		WeightTransformer wt = new WeightTransformer(weights).fit(iris);
		assertTrue(MatUtils.equalsExactly(iris, wt.transform(iris)));
		
		// assert on 0.0 all 0.0
		weights = VecUtils.rep(0.0, 4);
		wt = new WeightTransformer(weights).fit(iris);
		assertTrue(MatUtils.equalsExactly(new Array2DRowRealMatrix(new double[150][4],false), wt.transform(iris)));
	
		// assert that inv transform will create a matrix entirely of Infs...
		assertTrue(MatUtils.equalsExactly(new Array2DRowRealMatrix(MatUtils.rep(Double.POSITIVE_INFINITY, 150, 4),false), wt.inverseTransform(iris)));
		
		// assert dim mismatch on the fit, trans and inv trans methods.
		boolean a = false;
		try { wt.fit(TestSuite.getRandom(2, 2)); }
		catch(DimensionMismatchException e) { a= true; }
		finally { assertTrue(a); }
		
		a = false;
		try { wt.transform(TestSuite.getRandom(2, 2)); }
		catch(DimensionMismatchException e) { a= true; }
		finally { assertTrue(a); }
		
		a = false;
		try { wt.inverseTransform(TestSuite.getRandom(2, 2)); }
		catch(DimensionMismatchException e) { a= true; }
		finally { assertTrue(a); }
	}
}
