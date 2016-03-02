package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KMeans;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.TableFormatter.Table;
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
	
	public static Table formatKernelMatrix(final double[][] data, final Kernel kernel) {
		return formatter.format(new Array2DRowRealMatrix(kernel.kernelSimilarityMatrixUT(data), false));
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
		final double sigma = 0.05;
		double[][] simMat;

		
		// Look at the kernel matrices...
		Kernel kernel = new LinearKernel();
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1,  3,  4},
			new double[]{3, 13, 16},
			new double[]{4, 16, 20}
		}));

		
		kernel = new LaplacianKernel(sigma);
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1.0, 0.8681234453945849, 0.8350384028320718},
			new double[]{0.8681234453945849, 1.0, 0.951229424500714 },
			new double[]{0.8350384028320718, 0.951229424500714 , 1.0}
		}));
		
		
		kernel = new ANOVAKernel(sigma, 1);
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2.0, 1.6374615061559636, 1.4563589046997552},
			new double[]{1.6374615061559636, 2.0, 1.951229424500714 },
			new double[]{1.4563589046997552, 1.951229424500714, 2.0 }
		}));
		
		
		kernel = new SplineKernel();
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2.3333333333333335, 5.333333333333333,  6.833333333333333  },
			new double[]{5.333333333333333 , 145.66666666666666, 203.16666666666666 },
			new double[]{6.833333333333333 , 203.16666666666666, 293.88888888888886 }
		}));
		
		
		kernel = new PolynomialKernel();
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2,  4,  5},
			new double[]{4, 14, 17},
			new double[]{5, 17, 21}
		}));
		
		
		kernel = new HyperbolicTangentKernel();
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{0.9640275800758169, 0.999329299739067,  0.9999092042625951 },
			new double[]{0.999329299739067 , 0.9999999999986171, 0.9999999999999966 },
			new double[]{0.9999092042625951 , 0.9999999999999966, 1.0 }
		}));
		
		kernel = new RadialBasisKernel(sigma);
		simMat = kernel.kernelSimilarityMatrixFull(train_array);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1.0, 0.6703200460356393, 0.522045776761016 },
			new double[]{0.6703200460356393, 1.0, 0.951229424500714 },
			new double[]{0.522045776761016, 0.951229424500714, 1.0  }
		}));
		
		
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
