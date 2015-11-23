package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids.KMedoidsPlanner;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MatrixFormatter;

public class ClustTests {
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
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
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
		KMeans km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(2).setScale(true)).fit();
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
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
		KMeans km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(3).setScale(false)).fit();
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
		assertTrue(km.getPredictedLabels()[0] != km.getPredictedLabels()[3]);
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
		KMeans km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(3).setScale(true)).fit();
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
		assertTrue(km.getPredictedLabels()[0] != km.getPredictedLabels()[3]);
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
			km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(1).setScale(b)).fit();
			assertTrue(km.didConverge());

			if(b)
				assertTrue(km.totalCost() == 9.0);
		}
		
		// Test predict function
		assertTrue(km.predict(new double[]{100d, 201d, 101d}) == km.getPredictedLabels()[3]);
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
			km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(2).setScale(b));
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
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
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
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
		assertTrue(km.getPredictedLabels()[0] == km.getPredictedLabels()[3]);
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
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
		assertTrue(km.getPredictedLabels()[0] != km.getPredictedLabels()[3]);
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
		
		assertTrue(km.getPredictedLabels()[1] == km.getPredictedLabels()[2]);
		assertTrue(km.getPredictedLabels()[0] != km.getPredictedLabels()[3]);
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
	
	
	/* Test KNN with k = 1, 2 and 3; scale = true and false */
	@Test
	public void KNNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002}
		};
		
		final double[][] test_array = new double[][] {
			new double[] {0.01302, 	 0.0012,    0.06948},
			new double[] {3.01837,   2.2293,    3.94812}
		};
		
		final int[] trainLabels = new int[] {0, 1, 1};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		final Array2DRowRealMatrix test  = new Array2DRowRealMatrix(test_array);
		
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,2};
		
		KNN knn = null;
		for(boolean b : scale) {
			for(int k : ks) {
				knn = new KNN(train, test, trainLabels, new KNN.KNNPlanner(k).setScale(b).setVerbose(!b));
				knn.fit();
				
				final int[] results = knn.getPredictedLabels();
				assertTrue(results[0] == trainLabels[0]);
				assertTrue(results[1] == trainLabels[1]);
			}
		}
		
		// Try with k = 3, labels will be 1 both ways:
		for(boolean b : scale) {
			// Only verbose if scaling just to avoid too many loggings from this one test
			knn = new KNN(train, test, trainLabels, new KNN.KNNPlanner(3).setScale(b));
			knn.fit();
			
			final int[] results = knn.getPredictedLabels();
			assertTrue(results[0] == trainLabels[1]);
			assertTrue(results[1] == trainLabels[1]);
		}
		

		//knn.info("testing the KNN logger");
	}
	
	@Test
	public void KMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(10000, 10);
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5,7};
		
		KMeans km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(k).setScale(b));
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
	public void agglomerativeTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		AgglomerativeClusterer agglom = new AgglomerativeClusterer(mat);
		agglom.fit();
		
		assertTrue(agglom.getTree().size() == mat.getRowDimension()*2-1);
		
		@SuppressWarnings("unused")
		String cluster = formatter.format(agglom.getTree().getRoot().rightChild().getCluster().toRealMatrix());
		//System.out.println(cluster);
		//System.out.println(agglom);
	}
	
	@Test
	public void agglomerativeTest2CornerCase() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		AgglomerativeClusterer agglom = new AgglomerativeClusterer(mat);
		agglom.fit();
		
		assertTrue(agglom.getTree().size() == mat.getRowDimension()*2-1);
		assertTrue(agglom.getTree().size() == 1);
		
		// Quick logging test
		//agglom.info("testing the agglom logger");
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void agglomerativeTest3CornerCase() {
		final double[][] train_array = new double[][] {};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		new AgglomerativeClusterer(mat);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void agglomerativeTest2() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		AgglomerativeClusterer agglom = new AgglomerativeClusterer(mat, 
				new AgglomerativeClusterer.BaseHierarchicalPlanner().setLinkage(null)); // Setting linkage to null will break switch
		agglom.fit();
	}
	
	@Test
	public void agglomerativeTestHuge() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		AgglomerativeClusterer agglom = new AgglomerativeClusterer(mat, 
				new AgglomerativeClusterer.BaseHierarchicalPlanner()
					.setScale(true)
					.setVerbose(true));
		agglom.fit();
		
		assertTrue(agglom.getTree().size() == mat.getRowDimension()*2-1);
	}
	
	@Test
	public void KMeansLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(5000, 10);
		KMeans km = new KMeans(mat, new KMeans
				.BaseKCentroidPlanner(5)
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
			.setMinPts(2)
			.setVerbose(true))
				.fit();
		
		assertTrue(db.getNumberOfIdentifiedClusters() > 0);
	}
}
