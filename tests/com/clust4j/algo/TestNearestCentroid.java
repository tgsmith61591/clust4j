package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.NearestCentroid.NearestCentroidPlanner;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class TestNearestCentroid {
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
	
	@Test(expected=IllegalArgumentException.class)
	public void testNCBasicIAE() {
		new NearestCentroid(data, new int[]{0,0,0});
	}
	
	@Test
	public void testWarning() {
		NearestCentroid n = new NearestCentroid(data, new int[]{0,1,1}, 
			new NearestCentroidPlanner()
				.setSep(new GaussianKernel()));
		assertTrue(n.hasWarnings());
		assertTrue(n.getSeparabilityMetric().equals(AbstractClusterer.DEF_DIST));
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
				.setSep(Distance.MANHATTAN)).fit();
		assertTrue(VecUtils.equalsExactly(n.predict(data), new int[]{212,56,56}));
	}
}
