package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids.KMedoidsPlanner;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.NearestNeighbors.RunMode;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.NaNException;
import com.clust4j.utils.VecUtils;

public class ClustTests {
	{ // initializer
		com.clust4j.GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = false;
	}
	
	private static boolean print = false;
	private static final MatrixFormatter formatter = new MatrixFormatter();
	public static Array2DRowRealMatrix getRandom(final int rows, final int cols) {
		final Random rand = new Random();
		final double[][] data = new double[rows][cols];
		
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				data[i][j] = rand.nextDouble() * (rand.nextDouble() > 0.5 ? -1 : 1);
		
		return new Array2DRowRealMatrix(data);
	}

	@Test
	public void testFormatter() {
		final double[][] data = new double[][] {
			new double[] {0.0128275, 0.182751, 0.1284},
			new double[] {0.65816,   1.29518,  2.123316},
			new double[] {4.1234,    0.0001,   1.000002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		if(print) System.out.println(formatter.format(mat));
	}

	@Test
	public void mutabilityTest1() {
		final double eps = 0.3;
		final Array2DRowRealMatrix mat = getRandom(5,5);
		final double val11 = mat.getEntry(0, 0);
		
		DBSCAN db1 = new DBSCAN(mat, eps); // No scaling
		DBSCAN db2 = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(eps).setScale(true));
		
		// Testing mutability of scaling
		assertTrue(db1.getData().getEntry(0, 0) == val11);
		assertFalse(db2.getData().getEntry(0, 0) == val11);
	}
	
	/** Scale = false */
	@Test
	public void KMeansTest1() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, 2).fit();
		
		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		//km.info("testing the kmeans logger");
	}
	
	/** Now scale = true */
	@Test
	public void KMeansTest2() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(2).setScale(true)).fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
	}
	
	
	/** Now scale = false and multiclass */
	@Test
	public void KMeansTest3() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(3).setScale(false)).fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = true and multiclass */
	@Test
	public void KMeansTest4() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(3).setScale(true)).fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	// What if k = 1??
	@Test
	public void KMeansTest5() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
			km = new KMeans(mat, new KMeans.KMeansPlanner(1).setScale(b)).fit();
			assertTrue(km.didConverge());

			System.out.println(Arrays.toString(km.getLabels()));
			System.out.println(km.totalCost());
			if(b)
				assertTrue(km.totalCost() == 9.0);
		}
		
		// Test predict function -- no longer part of API
		// assertTrue(km.predictCentroid(new double[]{100d, 201d, 101d}) == km.getLabels()[3]);
	}
	
	// Make sure it won't break on a tie...
	@Test
	public void KMeansTieTest() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
			km = new KMeans(mat, new KMeans.KMeansPlanner(2).setScale(b));
			km.fit();
		}
	}
	
	/** Scale = false */
	@Test
	public void KMedoidsTest1() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, 2);
		assertTrue(km.getSeparabilityMetric().equals(Distance.MANHATTAN));
		
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		//km.info("testing the k-medoids logger");
	}
	
	/** Scale = true */
	@Test
	public void KMedoidsTest2() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002},
			new double[] {0.015, 	 0.161352,  0.1173},
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, 
				new KMedoidsPlanner(2)
					.setScale(true)
					.setVerbose(true));
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1 && km.getLabels()[3] == 0);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] == km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = false and multiclass */
	@Test
	public void KMedoidsTest3() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, 
				new KMedoidsPlanner(3)
					.setScale(false)
					.setVerbose(true));
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1 && km.getLabels()[3] == 2);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = true and multiclass */
	@Test
	public void KMedoidsTest4() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.2801,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, new KMedoidsPlanner(3).setScale(true));
		km.fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	// What if k = 1??
	@Test
	public void KMedoidsTest5() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final boolean[] scale = new boolean[]{false, true};
		
		KMedoids km = null;
		for(boolean b : scale) {
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
			km = new KMedoids(mat, new KMedoidsPlanner(1).setScale(b));
			km.fit();
			assertTrue(km.didConverge());
		}
	}
	
	@Test
	public void KMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(10000, 10);
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5,7};
		
		KMeans km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMeans(mat, new KMeans.KMeansPlanner(k).setScale(b));
				km.fit();
			}
		}
	}
	
	@Test
	public void KMedoidsLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(1000, 10);
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5};
		
		KMedoids km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMedoids(mat, 
						new KMedoidsPlanner(k)
							.setScale(b) );
				km.fit();
			}
		}
	}
	
	@Test
	public void KMeansLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(5000, 10);
		KMeans km = new KMeans(mat, new KMeans
				.KMeansPlanner(5)
					.setScale(true)
					.setVerbose(true)
				);
		km.fit();
	}
	
	@Test
	public void KMedoidsLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		KMedoids km = new KMedoids(mat, 
				new KMedoidsPlanner(5)
					.setScale(true)
					.setVerbose(true)
				);
		km.fit();
	}
	
	@Test
	public void DBSCANTest1() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.05)
			.setScale(true)
			.setVerbose(true)).fit();
	}
	
	@Test
	public void DBSCANTest2() {
		final double[][] train_array = new double[][] {
				new double[] {0.00504, 	 0.0001,    0.08172},
				new double[] {3.65816,   2.9471,    3.12331},
				new double[] {4.12344,   3.0001,    2.89002}
			};
			
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.75)
			.setScale(true)
			.setMinPts(1)
			.setVerbose(true))
				.fit();
		
		assertTrue(db.getNumberOfIdentifiedClusters() > 0);
		assertTrue(db.getLabels()[1] == db.getLabels()[2]);
	}
	
	@Test
	public void DBSCANTest3() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002},
			new double[] {0.00403, 	 0.0003,    0.08231}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.75)
			.setScale(true)
			.setMinPts(1)
			.setVerbose(true))
				.fit();
		
		assertTrue(db.getNumberOfIdentifiedClusters() == 2);
		assertTrue(db.getLabels()[1] == db.getLabels()[2]);
		assertTrue(db.getLabels()[0] == db.getLabels()[3]);
		assertFalse(db.hasWarnings());
	}
	
	@Test
	public void DBSCANLoadTest() {
		try {
			final Array2DRowRealMatrix mat = getRandom(5000, 10);
			new DBSCAN(mat, new DBSCAN
				.DBSCANPlanner()
					.setVerbose(true))
				.fit();
		} catch(OutOfMemoryError | StackOverflowError e) {
			return; // Not enough heap space..
		}
	}
	
	@Test
	public void MeanShiftTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {5.0,  4.3,  19.0, 4.0},
			new double[] {9.06, 12.6, 3.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.5)
				.setVerbose(true)).fit();
		
		assertTrue(ms.getNumberOfIdentifiedClusters() == 3);
		assertTrue(ms.getNumberOfNoisePoints() == 0);
		assertFalse(ms.hasWarnings());
	}
	
	@Test
	public void MeanShiftTest2() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.5)
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 4);
		assertTrue(ms.getLabels()[2] == ms.getLabels()[3]);
		System.out.println();
		
		
		ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.05)
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 5);
		assertFalse(ms.hasWarnings());
	}
	
	@Test
	public void AffinityPropTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagation
						.AffinityPropagationPlanner()
							.useGaussianSmoothing(bool)
							.setVerbose(true)
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
					if(bool) assertTrue(a.getNumberOfIdentifiedClusters() == 3);
					assertTrue(a.didConverge());
					assertTrue(labels[0] == 0);
					assertTrue(labels[2] == 1);
					assertTrue(labels[4] == 2);
		}
	}
	
	@Test
	public void AffinityPropLoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1000, 10);
		new AffinityPropagation(mat, new AffinityPropagation
			.AffinityPropagationPlanner()
				.setVerbose(true)).fit();
	}
	
	@Test
	public void NNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		NearestNeighbors nn = new NearestNeighbors(mat, 
			new NearestNeighborsPlanner()
				.setVerbose(true)
				.setK(1)).fit();
		
		ArrayList<Integer>[] ne = nn.getNearest();
		assertTrue(ne[0].size() == 1);
		assertTrue(ne[0].get(0) == 1);
		System.out.println();
		
		nn = new NearestNeighbors(mat, 
			new NearestNeighborsPlanner(RunMode.RADIUS)
				.setVerbose(true)
				.setRadius(3d)).fit();
		
		ne = nn.getNearest();
		assertTrue(ne[0].size() == 1);
		assertTrue(ne[1].size() == 1);
		assertTrue(ne[2].isEmpty());
		
		assertTrue( VecUtils.equalsExactly(nn.getNearestRecords(0)[0],train_array[1]) );
		assertTrue( VecUtils.equalsExactly(nn.getNearestRecords(1)[0],train_array[0]) );
	}
	
	@Test
	public void NN_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighborsPlanner()
					.setVerbose(true)
					.setK(k)).fit();
		}
	}
	
	@Test
	public void NN_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new NearestNeighbors(mat, 
				new NearestNeighborsPlanner(RunMode.RADIUS)
					.setVerbose(true)
					.setRadius(radius)).fit();
			System.out.println();
		}
	}
	
	@Test(expected=NaNException.class)
	public void testNanException() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  Double.NaN,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		new NearestNeighbors(mat);
	}
	
	@Test
	public void testChunkSizeMeanShift() {
		final int chunkSize = 500;
		assertTrue(MeanShift.getNumChunks(chunkSize, 500) == 1);
		assertTrue(MeanShift.getNumChunks(chunkSize, 501) == 2);
		assertTrue(MeanShift.getNumChunks(chunkSize, 23) == 1);
		assertTrue(MeanShift.getNumChunks(chunkSize, 10) == 1);
	}
	
	@Test
	public void testMeanShiftAutoBwEstimate() {
		final double[][] x = new double[][]{
			new double[]{ 0.08594657,  0.60925865,  0.39881186,  0.26410921,  0.19803359, 0.21035565,  0.11966197,  0.52581139,  0.38387628,  0.4825036 },
			new double[]{ 0.56943745,  0.9438055 ,  0.03867595,  0.49143331,  0.27470736, 0.08862225,  0.6203588 ,  0.76004573,  0.28224907,  0.98504973},
			new double[]{ 0.83903022,  0.48852263,  0.06229877,  0.00903001,  0.63412978, 0.01088595,  0.34147105,  0.31239485,  0.54709824,  0.60489573},
			new double[]{ 0.73606295,  0.64216898,  0.04652937,  0.00361512,  0.26470677, 0.59203757,  0.6751856 ,  0.36033876,  0.91600272,  0.97278172},
			new double[]{ 0.58265159,  0.82568491,  0.18918761,  0.85330221,  0.12915527, 0.21483348,  0.4457545 ,  0.61225749,  0.59133551,  0.74850421},
			new double[]{ 0.57694678,  0.71203528,  0.93801156,  0.12880288,  0.37977797, 0.1621018 ,  0.28997765,  0.48443738,  0.22086107,  0.33982515},
			new double[]{ 0.44102127,  0.03442744,  0.85580514,  0.49294382,  0.49326928, 0.16050842,  0.07405442,  0.55629147,  0.24660354,  0.12663662},
			new double[]{ 0.23211021,  0.94979067,  0.05108022,  0.59183824,  0.14980919, 0.32508404,  0.59326028,  0.92051835,  0.05639324,  0.53309613},
			new double[]{ 0.04267053,  0.78979235,  0.67823901,  0.47411163,  0.47987525, 0.64574195,  0.19336916,  0.75687751,  0.62494332,  0.31120583},
			new double[]{ 0.35446975,  0.90690062,  0.84454885,  0.37504377,  0.30435096, 0.91211773,  0.00132654,  0.08953336,  0.77461863,  0.51186425}
		};
		
		double bw = MeanShift.autoEstimateBW(new Array2DRowRealMatrix(x, false), 0.3, Distance.EUCLIDEAN, new Random());
		System.out.println(bw);
		
		// TODO
	}
}
