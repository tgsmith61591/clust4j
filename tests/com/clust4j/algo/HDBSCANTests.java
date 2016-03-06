package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.TreeMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.HDBSCAN.Algorithm;
import com.clust4j.algo.HDBSCAN.HDBSCANPlanner;
import com.clust4j.algo.HDBSCAN.HList;
import com.clust4j.algo.HDBSCAN.LinkageTreeUtils;
import com.clust4j.algo.HDBSCAN.TreeUnionFind;
import com.clust4j.algo.HDBSCAN.UnionFind;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.Inequality;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.MatSeries;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.QuadTup;

public class HDBSCANTests implements ClusterTest, ClassifierTest, BaseModelTest {
	final Array2DRowRealMatrix DATA = ExampleDataSets.IRIS.getData();
	final static MatrixFormatter formatter = TestSuite.formatter;
	final static double[][] dist_mat = new double[][]{
		new double[]{1,2,3},
		new double[]{4,5,6},
		new double[]{7,8,9}
	};

	
	@Test
	public void testHDBSCANGenericMutualReachability() {
		final int m = dist_mat.length, minPts = 3;
		
		final int min_points = FastMath.min(m - 1, minPts);
		final double[] core_distances = MatUtils
			.sortColsAsc(dist_mat)[min_points];
		
		final MatSeries ser1 = new MatSeries(core_distances, Inequality.GT, dist_mat);
		double[][] stage1 = MatUtils.where(ser1, core_distances, dist_mat);
		
		stage1 = MatUtils.transpose(stage1);
		final MatSeries ser2 = new MatSeries(core_distances, Inequality.GT, stage1);
		final double[][] result = MatUtils.where(ser2, core_distances, stage1);
		
		final double[][] res = MatUtils.transpose(result);
		assertTrue(MatUtils.equalsExactly(res, new double[][]{
			new double[]{7,8,9},
			new double[]{8,8,9},
			new double[]{9,9,9}
		}));
	}
	
	@Test
	public void testHDBSCANGenericMstLinkageCore() {
		double[][] X = new double[][]{
			new double[]{ 0.1,   0.6,   0.3},
			new double[]{ 0.6,   0.6,   0.6},
			new double[]{12.1,  13.1,  11.8}
		};
		
		final int m = X.length;
		
		double[][] result = HDBSCAN.LinkageTreeUtils.minSpanTreeLinkageCore(X, m);
		
		final double[][] expected = new double[][]{
			new double[]{0.0, 2.0, 0.3},
			new double[]{2.0, 1.0, 0.6}
		};
		
		assertTrue(MatUtils.equalsExactly(result, expected));
	}
	
	@Test
	public void testArgSorting() {
		double[][] m = new double[][]{
			new double[]{0.0 ,  2.0 ,  0.3},
			new double[]{2.0 ,  1.0 ,  0.6}
		};
		
		double[][] exp = MatUtils.copy(m);
		int[] sortedArgs = VecUtils.argSort(MatUtils.transpose(m)[2]);
		m = MatUtils.reorder(m, sortedArgs);
		
		assertTrue(MatUtils.equalsExactly(exp, m));
	}
	
	@Test
	public void testLabeling() {
		double[][] x = new double[][]{
			new double[]{0.0, 2.0, 0.3},
			new double[]{2.0, 1.0, 0.6},
		};
		
		double[][] y = new double[][]{
			new double[]{0.0, 2.0, 0.3, 2.0},
			new double[]{3.0, 1.0, 0.6, 3.0}
		};
		
		assertTrue(MatUtils.equalsExactly(HDBSCAN.label(x), y));
		
		// More complex
		assertTrue(MatUtils.equalsExactly(HDBSCAN.label(
				new double[][]{
					new double[]{0.0,2.0,0.3,1.9}, 
					new double[]{2.0,1.0,0.6,6.7},
					new double[]{1.0,4.3,0.9,0.1}
				}), 
				new double[][]{
			new double[]{0.0,2.0,0.3,2.0,0.0},
		    new double[]{4.0,1.0,0.6,3.0,0.0},
		    new double[]{5.0,5.0,0.9,6.0,0.}
		}));
	}
	
	@Test
	public void testCondenseAndComputeStability() {
		double[][] slt = new double[][]{
			new double[]{0.0, 2.0, 0.3, 2.0},
			new double[]{3.0, 1.0, 0.6, 3.0}
		};
		
		HList<QuadTup<Integer, Integer, Double, Integer>> h = HDBSCAN.LinkageTreeUtils.condenseTree(slt, 5);
		QuadTup<Integer, Integer, Double, Integer> q = h.get(0);
		assertTrue(q.one == 3);
		assertTrue(q.two == 0);
		// Three is a repeating decimal...
		assertTrue(q.four == 1);
		
		TreeMap<Integer, Double> computedStability = HDBSCAN.LinkageTreeUtils.computeStability(h);
		assertTrue(computedStability.size() == 1);
		assertTrue(computedStability.get(3) == 5);
		
		int[] labels = HDBSCAN.getLabels(h, computedStability);
		assertTrue(labels.length == 3);
		assertTrue(labels[0] == -1);
		assertTrue(labels[1] == -1);
		assertTrue(labels[2] == -1);
	}
	
	@Test
	public void testUnionFindClass() {
		UnionFind u = new UnionFind(5);
		int val = 2;
		
		// Test finds
		assertTrue(VecUtils.equalsExactly(new int[]{1,1,1,1,1,0,0,0,0}, u.size));
		assertTrue(u.fastFind(val) == val);
		assertTrue(u.parent[u.parent.length - 1] == val);
		assertFalse(u.parent[val] == val); // Should look like [-1, -1, -1, -1, -1, -1, -1, -1,  2]
		assertTrue(u.fastFind(-1) == val);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1, -1, -1, -1, -1, -1, 2}));
		assertTrue(u.fastFind(3) == 3);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1, -1, -1, -1, -1, -1, 3}));
		
		// Test unions
		u.union(3, 4);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1, -1, 3}));
		assertTrue(VecUtils.equalsExactly(u.size, new int[]{1, 1, 1, 1, 1, 2, 0, 0, 0}));
		
		u.union(-1, -2);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1,  6,  6}));
		
		assertTrue(u.find(6) == 6);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1,  6,  6}));
	}
	
	@Test
	public void testTreeUnionFindClass() {
		TreeUnionFind t = new TreeUnionFind(5);
		int[][] parent = new int[][]{
				new int[]{0,0},
				new int[]{1,0},
				new int[]{2,0},
				new int[]{3,0},
				new int[]{4,0}
			};
		
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		assertTrue(t.find(3) == 3);
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		
		assertTrue(t.find(-1) == 4);
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		
		// Test union
		t.union(3, -1);
		assertTrue(MatUtils.equalsExactly(t.dataArr, new int[][]{
			new int[]{0,0},
			new int[]{1,0},
			new int[]{2,0},
			new int[]{3,1},
			new int[]{3,0}
		}));
		
		t.union(-3, 4);
		assertTrue(MatUtils.equalsExactly(t.dataArr, new int[][]{
			new int[]{0,0},
			new int[]{1,0},
			new int[]{3,0},
			new int[]{3,1},
			new int[]{3,0}
		}));
		
		assertTrue(VecUtils.equalsExactly(t.components(), new int[]{0,1,3}));
	}

	@Test
	public void testGenericRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
				.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testFindNodeSplitDim() {
		final double[][] a = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		assertTrue(NearestNeighborHeapSearch.findNodeSplitDim(a, new int[]{0,1,2}) == 2);
	}
	
	
	@Test
	public void testPrimKD() {
		double m = dist_mat.length;
		int min_points = (int)FastMath.min(m - 1, 5);
		Array2DRowRealMatrix X = new Array2DRowRealMatrix(dist_mat);
		
		KDTree tree = new KDTree(X, HDBSCAN.DEF_LEAF_SIZE, Distance.EUCLIDEAN);
		Neighborhood query = tree.query(dist_mat, min_points, true, true);
		double[][] dists = query.getDistances();
		double[] coreDistances = MatUtils.getColumn(dists, dists[0].length - 1);
		
		// Needs to equal this:
		// [ 5.19615242,  5.19615242,  5.19615242]
		assertTrue(VecUtils.equalsExactly(coreDistances, new double[]{
			5.196152422706632, 5.196152422706632, 5.196152422706632
		}));
		
		assertTrue(MatUtils.equalsExactly(query.getValue(), new int[][]{
			new int[]{0,1},
			new int[]{1,0},
			new int[]{2,1}
		}));
		
		
		double[][] minSpanningTree = LinkageTreeUtils
				.minSpanTreeLinkageCore_cdist(dist_mat, coreDistances, 
						Distance.EUCLIDEAN, HDBSCAN.DEF_ALPHA);
		
		double[][] expected = new double[][]{
			new double[]{0.0, 1.0, 5.196152422706632},
			new double[]{1.0, 2.0, 5.196152422706632}
		};
		
		assertTrue(MatUtils.equalsExactly(minSpanningTree, expected));
	}
	
	@Test
	public void testPrimKDRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.PRIMS_KDTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testPrimBall() {
		double m = dist_mat.length;
		int min_points = (int)FastMath.min(m - 1, 5);
		Array2DRowRealMatrix X = new Array2DRowRealMatrix(dist_mat);
		
		BallTree tree = new BallTree(X, HDBSCAN.DEF_LEAF_SIZE, Distance.EUCLIDEAN);
		Neighborhood query = tree.query(dist_mat, min_points, true, true);
		double[][] dists = query.getDistances();
		double[] coreDistances = MatUtils.getColumn(dists, dists[0].length - 1);
		
		// Needs to equal this:
		// [ 5.19615242,  5.19615242,  5.19615242]
		assertTrue(VecUtils.equalsExactly(coreDistances, new double[]{
			5.196152422706632, 5.196152422706632, 5.196152422706632
		}));
		
		assertTrue(MatUtils.equalsExactly(query.getValue(), new int[][]{
			new int[]{0,1},
			new int[]{1,0},
			new int[]{2,1}
		}));
		
		
		double[][] minSpanningTree = LinkageTreeUtils
				.minSpanTreeLinkageCore_cdist(dist_mat, coreDistances, 
						Distance.EUCLIDEAN, HDBSCAN.DEF_ALPHA);
		
		double[][] expected = new double[][]{
			new double[]{0.0, 1.0, 5.196152422706632},
			new double[]{1.0, 2.0, 5.196152422706632}
		};
		
		assertTrue(MatUtils.equalsExactly(minSpanningTree, expected));
	}
	
	@Test
	public void testMoreBFS() {
		double[][] x = new double[][]{
			new double[]{0,1,1.414,2},
			new double[]{3,2,10.05,3}
		};
		
		HList<Integer> result;
		int root;
		
		// Test with root == 0
		root = 0;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == 1
		root = 1;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);

		// Test with root == 2
		root = 2;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == -1
		root = -1;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == -2
		root = -2;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
	}
	
	@Test
	public void testPrimBallRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.PRIMS_BALLTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testDescKeySet() {
		TreeMap<Integer, Double> stability = new TreeMap<>();
		stability.put(1, 456.0);
		stability.put(9, 23.0);
		stability.put(-5, 89.0);
			
		HList<Integer> nodes = HDBSCAN.GetLabelUtils.descSortedKeySet(stability);
		assertTrue(nodes.size() == 2);
		assertTrue(nodes.get(0) == 9);
		assertTrue(nodes.get(1) == 1);
		// It should trim the last one
	}
	
	@Test
	public void testSizeOverOne() {
		HList<QuadTup<Integer, Integer, Double, Integer>> tup = new HList<>();
		tup.add(new QuadTup<Integer, Integer, Double, Integer>(1,2,1.0,1));
		tup.add(new QuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		tup.add(new QuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		tup.add(new QuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		
		EntryPair<HList<double[]>, Integer> entry = 
			HDBSCAN.GetLabelUtils.childSizeGtOneAndMaxChild(tup);
		
		assertTrue(entry.getKey().size() == 3);
		assertTrue(entry.getValue() == 3);
	}
	
	@Test
	public void testDataSet() { // See if the iris dataset works...
		Array2DRowRealMatrix data = ExampleDataSets.IRIS.getData();
		new HDBSCAN(data, 
				new HDBSCANPlanner(1)
					.setVerbose(true)
					.setScale(true)).fit();
		
	}

	@Test
	@Override
	public void testScoring() {
		new HDBSCAN(DATA).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new HDBSCAN(DATA);
	}

	@Test
	@Override
	public void testArgConst() {
		new HDBSCAN(DATA, 3);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new HDBSCAN(DATA, new HDBSCANPlanner());
	}

	@Test
	@Override
	public void testFit() {
		new HDBSCAN(DATA, 1).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new HDBSCANPlanner().buildNewModelInstance(DATA);
		new HDBSCANPlanner(3).buildNewModelInstance(DATA);
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		HDBSCAN hd = new HDBSCAN(DATA, 
			new HDBSCAN.HDBSCANPlanner(1)
				.setVerbose(true)
				.setScale(true)).fit();
		System.out.println();

		final int[] labels = hd.getLabels();
		hd.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		HDBSCAN hd2 = (HDBSCAN)HDBSCAN.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(VecUtils.equalsExactly(hd2.getLabels(), labels));
		assertTrue(hd.equals(hd2));
		Files.delete(TestSuite.path);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testAlphaIAE() {
		new HDBSCAN(TestSuite.getRandom(5, 5), new HDBSCANPlanner().setAlpha(0.0));
	}
	
	@Test
	public void testSepWarn() {
		HDBSCAN h = new HDBSCAN(TestSuite.getRandom(5, 5), 
			new HDBSCANPlanner()
				.setAlgo(Algorithm.PRIMS_KDTREE)
				.setSep(new GaussianKernel()));
		assertTrue(h.hasWarnings());
	}
	
	@Test
	public void compareToPckg() {
		Array2DRowRealMatrix X= new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				0.58459246,  0.2411591 ,  0.54266953,  0.80298748,  0.7108317 ,
		        0.77419375,  0.19460038,  0.51769224,  0.80581355,  0.06109043,
		        0.57755264,  0.48690635,  0.4698578 ,  0.68256655,  0.35583625,
		        0.33956817,  0.46084149,  0.2266772 ,  0.78013553,  0.84169725,
		        0.45929076,  0.5763663 ,  0.85034392,  0.42344478,  0.08823549
			}, 5, 5), false);
		
		HDBSCAN h = new HDBSCAN(X).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		h = new HDBSCAN(X, new HDBSCANPlanner().setAlgo(Algorithm.PRIMS_KDTREE)).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		h = new HDBSCAN(X, new HDBSCANPlanner().setAlgo(Algorithm.PRIMS_BALLTREE)).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		// Test on IRIS
		X = ExampleDataSets.IRIS.getData();
		h = new HDBSCAN(X).fit();
		int[] expectedLabels = new int[]{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};
		
		assertTrue(VecUtils.equalsExactly(expectedLabels, h.getLabels()));
		
		// TODO fix KD & BALL trees
		h = new HDBSCAN(X, new HDBSCANPlanner().setAlgo(Algorithm.PRIMS_KDTREE)).fit();
		System.out.println(Arrays.toString(h.getLabels()));
	}
	
	@Test
	public void testMutualReachability() {
		Array2DRowRealMatrix X= new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		final double[][] dist = 
			ClustUtils.distanceFullMatrix(X, 
				Distance.EUCLIDEAN);
		
		
		// first test the partition by row
		final int minPts = FastMath.min(X.getRowDimension() - 1, 5);
		final double[] core_distances = MatUtils
				.sortColsAsc(dist)[minPts];
		assertTrue(VecUtils.equalsExactly(core_distances, new double[]{
			10.392304845413264,  5.196152422706632, 10.392304845413264
		}));
		
		final double[][] expected = new double[][]{
			new double[]{10.392304845413264, 10.392304845413264, 10.392304845413264},
			new double[]{10.392304845413264,  5.196152422706632, 10.392304845413264},
			new double[]{10.392304845413264, 10.392304845413264, 10.392304845413264},
		};
		
		double[][] mr = HDBSCAN.LinkageTreeUtils.mutualReachability(dist, minPts, 1.0);
		assertTrue(MatUtils.equalsExactly(mr, expected));
	}
	
	/*@Test
	public void testBoruvkaKDRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.BORUVKA_KDTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testBoruvkaBallRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.BORUVKA_BALLTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}*/
}
