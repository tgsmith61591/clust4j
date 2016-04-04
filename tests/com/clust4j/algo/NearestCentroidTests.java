package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.NearestCentroid.NearestCentroidPlanner;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class NearestCentroidTests implements ClassifierTest, ClusterTest, BaseModelTest {
	final Array2DRowRealMatrix data_ = ExampleDataSets.loadIris().getData();
	final int[] target_ = ExampleDataSets.loadIris().getLabels();

	@Test
	@Override
	public void testDefConst() {
		new NearestCentroid(data_, target_);
	}

	@Test
	@Override
	public void testArgConst() {
		// NA
		assertTrue(true);
		return;
	}

	@Test
	@Override
	public void testPlannerConst() {
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner());
	}

	@Test
	@Override
	public void testFit() {
		new NearestCentroid(data_, target_).fit();
		new NearestCentroid(data_, target_).fit().fit(); // Test fit again... ensure no exceptions
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner()).fit();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setShrinkage(0.5)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new NearestCentroid.NearestCentroidPlanner().buildNewModelInstance(data_, target_);
	}

	@Test
	@Override
	public void testScoring() {
		new NearestCentroid(data_, target_, new NearestCentroidPlanner().setVerbose(true)).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner()).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setVerbose(true)).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setShrinkage(0.5)).fit().score();
	}

	@Test(expected=DimensionMismatchException.class)
	public void testDME() {
		new NearestCentroid(data_, new int[]{1,2,3});
	}
	
	@Test
	public void testWarn() {
		/*// We need to allow this behavior now that NC used in KMeans
		NearestCentroid nn =
			new NearestCentroid(data_, target_, 
				new NearestCentroid.NearestCentroidPlanner()
					.setSep(new GaussianKernel()));
		assertTrue(nn.hasWarnings());
		*/
	}
	
	@Test
	public void testMiscellany() {
		assertTrue(new NearestCentroid.NearestCentroidPlanner()
			.getNormalizer().equals(AbstractClusterer.DEF_NORMALIZER));
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE1() {
		new NearestCentroid(data_, target_).getCentroids();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE2() {
		new NearestCentroid(data_, target_).predict(data_);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE3() {
		new NearestCentroid(data_, target_).getLabels();
	}
	
	@Test
	public void testLabels() {
		final int[] copy = new NearestCentroid(data_, target_).getTrainingLabels();
		copy[0] = 9; // Testing immutability of copy
		assertFalse(target_[0] == 9);
	}
	
	@Test
	public void testGetters() {
		NearestCentroid nn = new NearestCentroid(data_, target_).fit();
		nn.getCentroids();
		nn.predict(data_);
		nn.getLabels();
	}
	
	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		NearestCentroid nn = new NearestCentroid(data_, target_,
			new NearestCentroid.NearestCentroidPlanner()
				.setVerbose(true)
				.setScale(true)).fit();
		
		final int[] c = nn.getLabels();
		nn.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		NearestCentroid nn2 = (NearestCentroid)NearestCentroid.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(VecUtils.equalsExactly(c, nn2.getLabels()));
		assertTrue(nn2.equals(nn));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testCentroidViabilityKMeans() {
		final double[][] X = new double[][]{
			new double[]{0,0,0},
			new double[]{4,4,4},
			new double[]{8,8,8}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(X, false);
		NearestCentroid nn = new NearestCentroid(mat, new int[]{0,1,2},
			new NearestCentroid.NearestCentroidPlanner()
				.setVerbose(true)
				.setScale(false)).fit();
		
		Array2DRowRealMatrix Y = new Array2DRowRealMatrix(
			new double[][]{
				new double[]{0,0,0},
				new double[]{1,1,1},
				new double[]{4,4,4},
				new double[]{5,5,5},
				new double[]{8,8,8},
				new double[]{9,9,9}
			}, false);
		
		assertTrue(VecUtils.equalsExactly(nn.predict(Y), new int[]{0,0,1,1,2,2}));
	}
	
	private static final Array2DRowRealMatrix data = 
			new Array2DRowRealMatrix(new double[][]{
				new double[]{1,2,3},
				new double[]{4,5,6},
				new double[]{7,8,9}
			}, false);

	@Test
	public void testVarianceSqrtMedAdd() {
		double[][] x = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		ArrayList<double[]> cents = new ArrayList<double[]>();
		cents.add(new double[]{0.5, 0.5, 0.8});
		cents.add(new double[]{6d, 6d, 7d});
		
		int[] labs = new int[]{0,1,1};
		double[] variance = NearestCentroid.variance(x, cents, labs);
		assertTrue(VecUtils.equalsExactly(variance, 
			new double[]{ 5.25, 7.25, 9.84 }));
		
		int m = 3; int n = 2;
		double[] s  = NearestCentroid.sqrtMedAdd(variance, m, n);
		assertTrue(VecUtils.equalsExactly(s, 
			new double[]{ 4.983870251045172, 5.385164807134504, 5.829459831838877 }));
	}
	
	@Test
	public void testGetMSOuterProd() {
		double[] m = new double[]{1.15470054,  0.91287093};
		double[] s = new double[]{4.98387025,  5.38516481,  5.82945983};
		double[][] ms = NearestCentroid.mmsOuterProd(m, s);
		double[][] expected = new double[][]{
			new double[]{5.7548776689649355,  6.218252714095998,  6.731280413609309},
			new double[]{4.5496302701168325,  4.915960408307973,  5.321544416409742}
		};
		
		assertTrue(MatUtils.equalsExactly(ms, expected));
	}
	
	@Test
	public void testEm() {
		int[] nk = new int[]{1,2};
		int m = 3;
		
		// determine deviation
		double[] em = NearestCentroid.getMVec(nk, m);
		
		assertTrue(VecUtils.equalsExactly(em, 
			new double[]{1.1547005383792515, 0.9128709291752768}));
	}

	@Test
	public void testDev() {
		ArrayList<double[]> cents = new ArrayList<>();
		cents.add(new double[]{0.5,0.5,0.8});
		cents.add(new double[]{6.0,6.0,7.0});
		
		double[] cent = new double[]{3.0,3.0,4.5};
		double[][] ms = new double[][]{
			new double[]{5.7548776689649355,  6.218252714095998,  6.731280413609309},
			new double[]{4.5496302701168325,  4.915960408307973,  5.321544416409742}
		};
		
		double shrinkage = 0.5;
		
		double[][] shrunk = NearestCentroid.getDeviationMinShrink(cents, cent, ms, shrinkage);
		double[][] expected = new double[][]{
			new double[]{-0.0, -0.0, -0.3343597931953453},
			new double[]{0.725184864941584, 0.5420197958460131, 0.0}
		};
		
		assertTrue(MatUtils.equalsExactly(shrunk, expected));
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNCBasicException1() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1});
		n.getCentroids();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1});
		n.predict(data);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNCBasicException2() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1});
		n.getLabels();
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testNCBasicDimMismatch() {
		new NearestCentroid(data, new int[]{0,1,1,2});
	}
	
	@Test(expected=NullPointerException.class)
	public void testNPE() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1}, 
			new NearestCentroidPlanner()
				.setNormalizer(null)
				.setSeed(new Random())
				.setScale(true)
				.setVerbose(false));
		assertTrue(n.hasWarnings());
		assertTrue(n.getSeparabilityMetric().equals(AbstractClusterer.DEF_DIST));
	}
	
	@Test
	public void testBasicFit() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1}, 
			new NearestCentroidPlanner()
				.setShrinkage(null)).fit();
		final ArrayList<double[]> cents = n.getCentroids();
		
		assertTrue(cents.size() == 2);
		assertTrue(VecUtils.equalsExactly(cents.get(0), new double[]{1.0,2.0,3.0}));
		assertTrue(VecUtils.equalsExactly(cents.get(1), new double[]{5.5,6.5,7.5}));
		assertTrue(VecUtils.equalsExactly(n.predict(data), new int[]{0,1,1}));
	}
	
	@Test
	public void testShrinkageFit() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1}, 
			new NearestCentroidPlanner()
				.setShrinkage(0.5)
				.setVerbose(true)).fit();
		final ArrayList<double[]> cents = n.getCentroids();
		
		assertTrue(cents.size() == 2);
		assertTrue(VecUtils.equalsExactly(cents.get(0), new double[]{3.449489742783178, 4.449489742783178, 5.449489742783178}));
		assertTrue(VecUtils.equalsExactly(cents.get(1), new double[]{4.0,5.0,6.0}));
		assertTrue(VecUtils.equalsExactly(n.predict(data), new int[]{0,1,1}));
	}
	
	@Test
	public void testOddLabels() {
		NearestCentroid n = new NearestCentroid(data, new int[]{212,56,56}, 
			new NearestCentroidPlanner()).fit();
		assertTrue(VecUtils.equalsExactly(n.predict(data), new int[]{212,56,56}));
	}
	
	@Test
	public void testOddLabelsFromNewInstance() {
		NearestCentroid n = new NearestCentroidPlanner()
			.buildNewModelInstance(data, new int[]{-6,0,0})
			.fit();
		
		// Test fitting it again, ensure it returns immediately
		n.fit();
		
		assertTrue(VecUtils.equalsExactly(n.getTrainingLabels(), new int[]{-6,0,0}));
	}
	
	@Test
	public void testOddLabelsManhattan() {
		NearestCentroid n = new NearestCentroid(data, new int[]{212,56,56}, 
			new NearestCentroidPlanner()
				.setSep(Distance.MANHATTAN)
				.setVerbose(true)).fit();
		
		assertTrue(VecUtils.equalsExactly(n.predict(data), new int[]{212,56,56}));
	}
	
	@Test
	public void testLoadWithSingleClass() {
		NearestCentroid n = new NearestCentroid(TestSuite.getRandom(1200, 10), 
				VecUtils.repInt(1, 1200)).fit();
		assertTrue(VecUtils.equalsExactly(VecUtils.repInt(1, 5), n.predict(TestSuite.getRandom(5, 10))));
	}
	
	/**
	 * Asser that when all of the matrix entries are exactly the same,
	 * the algorithm will still converge, yet produce one label: 0
	 */
	@Override
	@Test
	public void testAllSame() {
		final double[][] x = MatUtils.rep(-1, 3, 3);
		final Array2DRowRealMatrix X = new Array2DRowRealMatrix(x, false);
		
		int[] labels = new NearestCentroid(X, new int[]{0,1,2}, new NearestCentroidPlanner().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.VecIntSeries(labels, Inequality.EQUAL_TO, 0).all());
		System.out.println();
		
		labels = new NearestCentroid(X, new int[]{0,1,2}, new NearestCentroidPlanner().setVerbose(true)).fit().predict(X);
		assertTrue(new VecUtils.VecIntSeries(labels, Inequality.EQUAL_TO, 0).all());
		System.out.println();
	}
}
