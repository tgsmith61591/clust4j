package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KMeans;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;

public class KernelTestCases {
	
	final static MatrixFormatter formatter = new MatrixFormatter();
	final static Random rand = new Random();
	
	public static double[] randomVector(int length) {
		final double[] a = new double[length];
		for(int i = 0; i < a.length; i++)
			a[i] = rand.nextDouble();
		return a;
	}
	
	public static String formatKernelMatrix(final double[][] data, final Kernel kernel) {
		return formatter.format(new Array2DRowRealMatrix(kernel.kernelSimilarityMatrix(data), false));
	}
	
	public static void print(final String s) {
		System.out.println(s);
	}

	@Test
	public void testSmall() {
		final double[] a = new double[]{0,1};
		final double[] b = new double[]{1,0};
		
		// Perfectly orthogonal
		assertTrue(new LinearKernel().getSimilarity(a, b) == new LinearKernel().getConstant());
		assertTrue(VecUtils.isOrthogonalTo(a, b));
	}
	
	@Test
	public void testProjections() {
		final double[] a = new double[]{5,0};
		final double[] b = new double[]{3,0};
		assertTrue(new LinearKernel().getSimilarity(a, b) == 15 + new LinearKernel().getConstant());
	}
	
	@Test
	public void testLinearSeparability() {
		// Perfectly linearly separable
		final double[][] train_array = new double[][] {
			new double[] {0.0, 	 1.0},
			new double[] {2.0,   3.0},
			new double[] {2.0,   4.0}
		};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		
		// Look at the kernel matrix...
		Kernel kernel = new LinearKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 3.0);

		final double sigma = 0.05;
		
		kernel = new LaplacianKernel(sigma);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.8681234453945849);
		
		kernel = new ANOVAKernel(sigma, 1);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 1.6374615061559636);
		
		kernel = new SplineKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 5.333333333333333);
		
		kernel = new PolynomialKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 4.0);
		
		kernel = new HyperbolicTangentKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.999329299739067);
		
		kernel = new RadialBasisKernel(sigma);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.6703200460356393);
		
		
		
		// Test with KMEANS
		KMeans km = new KMeans(train, 
				new KMeans.KMeansPlanner(2)
					.setSep(kernel)
					.setVerbose(true)
				);
		km.fit();
		System.out.println();
	}
	
}
