package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.algo.BallTree;
import com.clust4j.algo.KDTree;
import com.clust4j.algo.NearestNeighborHeapSearch.Heap;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeHeap.NodeHeapData;
import com.clust4j.algo.NearestNeighborHeapSearch.MutableDouble;
import com.clust4j.algo.NearestNeighborHeapSearch.NeighborsHeap;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeData;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeHeap;
import com.clust4j.algo.NearestNeighborHeapSearch.PartialKernelDensity;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.HaversineDistance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.QuadTup;
import com.clust4j.utils.TriTup;
import com.clust4j.utils.VecUtils;

public class NNHSTests {
	final public static Array2DRowRealMatrix IRIS = ExampleDataSets.loadIris().getData();
	
	final static double[][] a = new double[][]{
		new double[]{0,1,0,2},
		new double[]{0,0,1,2},
		new double[]{5,6,7,4}
	};

	@Test
	public void testKD1() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		KDTree kd = new KDTree(mat);
		
		QuadTup<double[][], int[], NodeData[], double[][][]> arrays = kd.getArrays();
		
		assertTrue(MatUtils.equalsExactly(arrays.one, a));
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2}, arrays.two));
		
		TriTup<Integer, Integer, Integer> stats = kd.getTreeStats();
		assertTrue(stats.one == 0);
		assertTrue(stats.two == 0);
		assertTrue(stats.three==0);
		
		NodeData data = arrays.three[0];
		assertTrue(data.idx_start == 0);
		assertTrue(data.idx_end == 3);
		assertTrue(data.is_leaf);
		assertTrue(data.radius == 1);
	}
	
	@Test
	public void testBall1() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		BallTree ball = new BallTree(mat);
		
		QuadTup<double[][], int[], NodeData[], double[][][]> arrays = ball.getArrays();
		
		assertTrue(MatUtils.equalsExactly(arrays.one, a));
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2}, arrays.two));
		
		TriTup<Integer, Integer, Integer> stats = ball.getTreeStats();
		assertTrue(stats.one == 0);
		assertTrue(stats.two == 0);
		assertTrue(stats.three==0);
		
		NodeData data = arrays.three[0];
		assertTrue(data.idx_start == 0);
		assertTrue(data.idx_end == 3);
		assertTrue(data.is_leaf);
		assertTrue(data.radius == 6.716480559869961);
		
		double[][][] trip = arrays.four;
		assertTrue(trip.length == 1);
		assertTrue(trip[0][0][0] == 1.6666666666666667);
		assertTrue(trip[0][0][1] == 2.3333333333333333);
		assertTrue(trip[0][0][2] == 2.6666666666666667);
		assertTrue(trip[0][0][3] == 2.6666666666666667);
	}

	
	@Test
	public void testKernelDensities() {
		// Test where dist > h first
		double dist = 5.0, h = 1.3;
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -7.396449704142011);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -3.846153846153846);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		
		// Test where dist < h second
		dist = 1.3; 
		h = 5.0;
		
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -0.033800000000000004);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == 0.0);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == -0.06999337182053497);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -0.26);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == -0.3011050927839216);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == -0.08582521637384073);
	}
	
	
	
	// ================== constructor tests
	@Test
	public void testConst1() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		Loggable log = null;
		
		// test kd constructors
		KDTree kd = new KDTree(A);
		kd = new KDTree(A, 5);
		kd = new KDTree(A, 5, Distance.EUCLIDEAN);
		assertTrue(kd.getLeafSize() == 5);
		kd = new KDTree(A, Distance.EUCLIDEAN);
		kd = new KDTree(A, log);
		assertTrue(kd.logger == null);
		kd = new KDTree(A, 5, Distance.EUCLIDEAN, null);
		
		
		BallTree ball = new BallTree(A);
		ball = new BallTree(A, 5);
		ball = new BallTree(A, 5, Distance.EUCLIDEAN);
		assertTrue(5 == ball.getLeafSize());
		ball = new BallTree(A, Distance.EUCLIDEAN);
		ball = new BallTree(A, log);
		assertTrue(ball.logger == null);
		ball = new BallTree(A, 5, Distance.EUCLIDEAN, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstIAE1() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		new KDTree(A, 0);
	}
	
	// Create anonymous DistanceMetric class to test
	@Test
	public void testConst2() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A, new DistanceMetric() {
			private static final long serialVersionUID = 6792348831585297421L;

			@Override public double getDistance(final double[] a, final double[] b) { return 0.0; }
			@Override public double getP() { return 0.0; }
			@Override public double getPartialDistance(final double[] a, final double[] b) { return getDistance(a, b); }
			@Override public double partialDistanceToDistance(double d) { return d; }
			@Override public double distanceToPartialDistance(double d) { return d; }
			@Override public String getName() { return "Test anonymous DistanceMetric"; }
		});
		
		assertTrue(kd.getMetric().equals(Distance.EUCLIDEAN));
	}
	
	private static void passByRef(MutableDouble md, double x) {
		md.value = x;
	}
	
	@Test
	public void testMutableDouble() {
		MutableDouble md = new MutableDouble(145d);
		passByRef(md, 15d);
		assertTrue(md.value == 15d);
		assertTrue(md.compareTo(14d) == 1);
		assertTrue(new MutableDouble().value == 0d);
	}
	
	@Test
	public void testNodeDataContainerClass() {
		// Test def constructor
		NodeData node = new NodeData();
		assertTrue(node.idx_start == 0);
		assertTrue(node.idx_end == 0);
		assertTrue(!node.is_leaf);
		assertTrue(node.radius == 0.0);
		
		// Test arg constructor
		node = new NodeData(1,2,true,5.9);
		assertTrue(node.idx_start == 1);
		assertTrue(node.idx_end == 2);
		assertTrue(node.is_leaf);
		assertTrue(node.radius == 5.9);
		
		// Test immutability
		NodeData node2 = node.copy();
		node2.idx_start = 15;
		node2.idx_end = 67;
		node2.is_leaf = false;
		node2.radius = 5.6;
		assertTrue(node.start() == 1);
		assertTrue(node.end() == 2);
		assertTrue(node.isLeaf());
		assertTrue(node.radius() == 5.9);
		
		// ensure won't throw exception
		node.toString();
	}
	
	@Test
	public void testGetterRefMutability() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		
		double[][] data = kd.getData();
		double[][] dataRef = kd.getDataRef();
		dataRef[0][0] = 150d;
		assertFalse(MatUtils.equalsExactly(kd.getDataRef(), data));
		
		double[][][] bounds = kd.getNodeBounds();
		double[][][] boundsRef = kd.getNodeBoundsRef();
		boundsRef[0][0][0] = 150;
		assertFalse(MatUtils.equalsExactly(kd.getNodeBoundsRef()[0], bounds[0]));
		
		int[] idcs = kd.getIndexArray();
		int[] idcsRef = kd.getIndexArrayRef();
		idcsRef[0] = 150;
		assertFalse(VecUtils.equalsExactly(kd.getIndexArrayRef(), idcs));
		
		NodeData[] nodes = kd.getNodeData();
		NodeData[] nodeRef=kd.getNodeDataRef();
		nodeRef[0].idx_end = 150;
		assertFalse(kd.getNodeDataRef()[0].idx_end == nodes[0].idx_end);
	}
	
	@Test
	public void testInstanceMethod() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		
		double[] b = new double[]{0,1,2};
		double[] c = new double[]{3,4,5};
		assertTrue(kd.dist(b,c) == Distance.EUCLIDEAN.getDistance(b, c));
		assertTrue(kd.rDist(b,c) == Distance.EUCLIDEAN.getPartialDistance(b, c));
		assertTrue(kd.rDistToDist(kd.rDist(b, c)) == Distance.EUCLIDEAN.partialDistanceToDistance(kd.rDist(b, c)));
		assertTrue(kd.getNumCalls() == 4);
		kd.resetNumCalls();
		
		assertTrue(kd.getNumCalls() == 0);
	}
	
	@Test
	public void testNodeFind() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		final int findNode = KDTree.findNodeSplitDim(a, kd.idx_array);
		assertTrue(findNode == 2);
	}
	
	@Test
	public void testKernels() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		double[] density;
		double bw = 0.5;
		
		// Ensure no exceptions in kernels
		for(PartialKernelDensity kern: PartialKernelDensity.values()) {
			density = kd.kernelDensity(a, bw, kern);
			System.out.println(Arrays.toString(density));
		}
	}
	
	@Test
	public void testSwap() {
		int[] ex = new int[]{0,1,2};
		KDTree.swap(ex, 0, 1);
		assertTrue(VecUtils.equalsExactly(ex, new int[]{1,0,2}));
	}
	
	@Test(expected=IllegalStateException.class)
	public void testNodeHeap1() {
		NodeHeap nh1 = new NodeHeap(0);
		assertTrue(nh1.data.length == 1); // picks max (size, 1)
		nh1 = new NodeHeap(2);
		assertTrue(nh1.data.length == 2);
		assertTrue(nh1.n == 0);
		nh1.clear();
		assertTrue(nh1.n == 0);
		assertTrue(null == nh1.peek());
		nh1.pop(); // throws the exception on empty heap
	}
	
	@Test
	public void testNodeHeapPushesPops() {
		NodeHeap heap = new NodeHeap(3);
		
		NodeHeapData h = new NodeHeapData(1.0,  0, 0);
		heap.push(new NodeHeapData(12.0, 1, 2));
		heap.push(new NodeHeapData(9.0,  4, 5));
		heap.push(new NodeHeapData(11.0, 9,-1));
		heap.push(h);
		
		assertTrue(heap.data.length == 8);
		assertTrue(heap.data[0].val == 1.0);
		assertTrue(heap.data[1].val == 9.0);
		assertTrue(heap.data[2].val ==11.0);
		assertTrue(heap.data[3].val ==12.0);
		
		assertTrue(heap.data[0].i1 == 0);
		assertTrue(heap.data[1].i1 == 4);
		assertTrue(heap.data[2].i1 == 9);
		assertTrue(heap.data[3].i1 == 1);
		
		assertTrue(heap.data[0].i2 == 0);
		assertTrue(heap.data[1].i2 == 5);
		assertTrue(heap.data[2].i2 ==-1);
		assertTrue(heap.data[3].i2 == 2);
		
		assertTrue(heap.data[0].equals(new NodeHeapData(1.0,0,0)));
		assertTrue(heap.data[0].equals(h));
		assertFalse(heap.data[0].equals(new Integer(1)));
		assertTrue(heap.n == 4);
		
		assertTrue(heap.pop().equals(h));
		assertTrue(heap.data[0].val == 9.0);
		assertTrue(heap.data[1].val ==12.0);
		assertTrue(heap.data[2].val ==11.0);
		assertTrue(null == heap.data[3]);
		
		// Ensure no NPE
		heap.toString();
	}
	
	@Test
	public void testDualSwap() {
		double[] a = new double[]{0,1,2};
		int[] b = new int[]{3,4,5};
		Heap.dualSwap(a, b, 0, 1);
		assertTrue(VecUtils.equalsExactly(a, new double[]{1,0,2}));
		assertTrue(VecUtils.equalsExactly(b, new int[]{4,3,5}));
	}
	
	@Test
	public void testBigKD() {
		Array2DRowRealMatrix x = new Array2DRowRealMatrix(IRIS.getData(),false);
		KDTree kd = new KDTree(x);
		assertTrue(VecUtils.equalsExactly(kd.idx_array, new int[]{
			0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
	        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
	        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
	        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  53,  57,
	        59,  60,  62,  64,  69,  71,  79,  80,  81,  82,  89,  92,  93,
	        98,  99,  88,  67,  61,  94,  95,  96,  74,  97,  90,  87,  65,
	        75, 106,  86,  68,  54,  55,  73,  91,  56,  63,  78,  51,  58,
	        66,  50,  84,  85, 138,  76,  70,  52, 121, 123, 126, 127,  72,
	       146,  77, 113, 119, 149, 109, 110, 111, 112, 100, 114, 115, 116,
	       117, 118, 101, 120, 102, 122, 103, 124, 125, 104, 105, 128, 129,
	       130, 131, 132, 133, 134, 135, 136, 137,  83, 139, 140, 141, 142,
	       143, 144, 145, 107, 147, 148, 108
		}));
	}
	
	@Test
	public void testQuerySmall() {
		KDTree kd = new KDTree(new Array2DRowRealMatrix(a));
		assertTrue(VecUtils.equalsExactly(kd.idx_array, new int[]{0,1,2}));
		assertTrue(kd.node_bounds.length == 2);
		assertTrue(kd.node_bounds[0].length == 1);
		assertTrue(kd.node_bounds[1].length == 1);
		assertTrue(VecUtils.equalsExactly(kd.node_bounds[0][0], new double[]{0.0,0.0,0.0,2.0}));
		assertTrue(VecUtils.equalsExactly(kd.node_bounds[1][0], new double[]{5.0,6.0,7.0,4.0}));
	
		
		double[][] expectedDists = new double[][]{ new double[]{0.0}, new double[]{0.0} };
		int[][] expectedIndices  = new int[][]{ new int[]{0}, new int[]{1} };
		
		Neighborhood neighb;
		boolean[] trueFalse = new boolean[]{true, false};
		for(boolean dualTree: trueFalse) {
			for(boolean sort: trueFalse) {
				neighb= new Neighborhood(
					kd.query(new double[][]{
						new double[]{0,1,0,2},
						new double[]{0,0,1,2}
					}, 1, dualTree, sort));
				
				assertTrue(MatUtils.equalsExactly(expectedDists, neighb.getDistances()));
				assertTrue(MatUtils.equalsExactly(expectedIndices, neighb.getIndices()));
			}
		}
	}
	
	@Test
	public void testSimultaneousSort() {
		double[] dists = new double[]{
			3.69675274,  2.89351805,  1.79065633,  
			0.44375205,  7.77409946,  7.08011014,
			8.41547227,  5.57512117,  8.85578907,
			2.60367035 };
		
		int[] indices = new int[]{
			4, 1, 0, 7, 6, 5, 8, 2, 3, 9
		};
		
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.44375205,  1.79065633,  2.60367035,
			2.89351805,  3.69675274,  5.57512117,
			7.08011014,  7.77409946,  8.41547227,
			8.85578907
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			7, 0, 9, 1, 4, 2, 5, 6, 8, 3
		}));
		
		
		
		
		dists = new double[]{0.7,0.1};
		indices = new int[]{2,1};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			1,2
		}));
		
		
		
		
		dists = new double[]{0.7};
		indices = new int[]{2};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			2
		}));
		
		
		
		dists = new double[]{0.7,0.1,0.3};
		indices = new int[]{2,1,3};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.3,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			1,3,2
		}));
		
		
		
		dists = new double[]{0.3,0.7,0.1};
		indices = new int[]{2,1,3};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.3,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			3,2,1
		}));
	}
	
	@Test
	public void testNeighborsHeap() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094},
			new double[]{-0.95392127,  0.72765662,  0.46466226},
			new double[]{ 0.57011545, -1.53581033,  0.52009414}
		};
		
		final int k = 1;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.48111094},
			new double[]{-0.95392127},
			new double[]{-1.53581033}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{2},
			new int[]{0},
			new int[]{1}
		}));
	}
	
	@Test
	public void testNeighborHeapOrderInPlace() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094,  0.0002354, 1.12345},
			new double[]{-0.95392127,  0.72765662,  0.46466226, -0.9128421, 5.12345},
			new double[]{ 0.57011545, -1.53581033,  0.52009414,  0.1958271, -4.3918}
		};
		
		final int k = 3;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.48111094, -0.26063195, 0.0002354},
			new double[]{-0.95392127, -0.9128421, 0.46466226},
			new double[]{-4.3918,     -1.53581033, 0.1958271}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{2,1,3},
			new int[]{0,3,2},
			new int[]{4,1,3}
		}));
	}
	
	@Test
	public void testNeighborHeapTwoAndLessLen() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195},
			new double[]{-0.95392127,  0.72765662},
			new double[]{ 0.57011545, -1.53581033}
		};
		
		
		
		int k = 1;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.26063195},
			new double[]{-0.95392127},
			new double[]{-1.53581033}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{1},
			new int[]{0},
			new int[]{1}
		}));
		
		
		
		k = 2;
		heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		neighb = new Neighborhood(heap.getArrays(true));
		
		dists = neighb.getDistances();
		inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.26063195, 0.15464338 },
			new double[]{-0.95392127, 0.72765662 },
			new double[]{-1.53581033, 0.57011545 }
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{1,0},
			new int[]{0,1},
			new int[]{1,0}
		}));
	}
	
	@Test
	public void testNeighborHeapNoSortAndLargest() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094,  0.0002354, 1.12345},
			new double[]{-0.95392127,  0.72765662,  0.46466226, -0.9128421, 5.12345},
			new double[]{ 0.57011545, -1.53581033,  0.52009414,  0.1958271, -4.3918}
		};
		
		final int k = 3;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(false));
		
		double[][] dists = neighb.getDistances();
		for(int row = 0; row < dists.length; row++)
			assertTrue(heap.largest(row) == VecUtils.max(dists[row]));
	}
	
	@Test
	public void testDistToRDist() {
		double[]a = new double[]{5,0,0};
		double[]b = new double[]{0,0,1};
		KDTree kd = new KDTree(IRIS);
		assertTrue(kd.dist(a, b) == 5.0990195135927845);
		assertTrue(kd.rDistToDist(25.999999999999996) == kd.dist(a, b));
		assertTrue(Precision.equals(kd.rDist(a, b), 25.999999999999996, 1e-8));
		assertTrue(Precision.equals(kd.rDistToDist(kd.rDist(a, b)), kd.dist(a, b), 1e-8));
	}
	
	@Test
	public void testMinRDistDual() {
		Array2DRowRealMatrix X1 = IRIS;
		
		double[][] query = new double[10][];
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		Array2DRowRealMatrix X2 = new Array2DRowRealMatrix(query, false);
		
		NearestNeighborHeapSearch tree1 = new KDTree(X1);
		NearestNeighborHeapSearch tree2 = new KDTree(X2);
		
		double dist = tree1.minRDistDual(tree1, 0, tree2, 0);
		assertTrue(0.0 == dist);
		dist = tree1.minRDistDual(tree1, 2, tree2, 0);
		assertTrue(7.930000000000001 == dist);
		
		
		tree1 = new BallTree(X1);
		tree2 = new BallTree(X2);
		
		dist = tree1.minRDistDual(tree1, 0, tree2, 0);
		assertTrue(0.0 == dist);
		
		dist = tree1.minRDistDual(tree1, 2, tree2, 0);
		// TODO: assertion
	}
	
	@Test
	public void testMinRDist() {
		Array2DRowRealMatrix X1 = IRIS;
		NearestNeighborHeapSearch tree1 = new KDTree(X1);
		double[] a = new double[]{5.1, 3.5, 1.4, 0.2};
		
		assertTrue(tree1.minRDist(tree1, 1, a) == 0);
		assertTrue(tree1.minRDist(tree1, 2, a) == 10.000000000000004);
		
		a = new double[]{4.9, 3.0, 1.4, 0.2};
		assertTrue(tree1.minRDist(tree1, 1, a) == 0);
		assertTrue(tree1.minRDist(tree1, 2, a) == 10.000000000000004);
	}
	
	@Test
	public void moreNodeHeapTests() {
		NodeHeap nh = new NodeHeap(10);
		assertTrue(nh.n == 0);
		
		nh.push(new NodeHeapData());
		assertTrue(nh.n == 1);
		
		nh.resize(15);
		assertTrue(nh.n == 1);
		
		nh.resize(2);
		assertTrue(nh.n == 1);
		
		
		// Now test some pushes...
		Random seed = new Random(5);
		NodeHeapData node;
		for(int i = 0; i < 10; i++) {
			
			node = new NodeHeapData(
				10.0 - i,
				//seed.nextDouble() * seed.nextInt(40),
				seed.nextInt(5),
				seed.nextInt(100)
			);
			
			nh.push(node);
		}
		
		assertTrue(nh.n == 11);
		nh.pop();
		assertTrue(nh.n == 10);
		assertTrue(nh.peek().val == 1.0);
		
		nh.toString(); // Ensure does not throw NPE
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void nodeHeapResizeUnder1() {
		NodeHeap nh = new NodeHeap(10);
		nh.resize(0); // Here is the exception
	}
	
	
	@Test
	public void testQueryBig() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedDists = new double[][]{
			new double[]{  0.        ,  0.1       ,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.24494897},
			new double[]{  0.        ,  0.14142136,  0.17320508},
			new double[]{  0.        ,  0.14142136,  0.17320508},
			new double[]{  0.        ,  0.33166248,  0.34641016},
			new double[]{  0.        ,  0.2236068 ,  0.26457513},
			new double[]{  0.        ,  0.1       ,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.3       },
			new double[]{  0.        ,  0.        ,  0.        }
		};
		
		int[][] expectedIndices = new int[][]{
			new int[]{ 0, 17,  4},
			new int[]{ 1, 45, 12},
			new int[]{ 2, 47,  3},
			new int[]{ 3, 47, 29},
			new int[]{ 4,  0, 17},
			new int[]{ 5, 18, 10},
			new int[]{ 6, 47,  2},
			new int[]{ 7, 39, 49},
			new int[]{ 8, 38,  3},
			new int[]{37,  9, 34}
		};
		

		// Assert node data equal
		NodeData[] expectedNodeData = new NodeData[]{
			new NodeData(0, 150, false, 10.29635857961444),
			new NodeData(0, 75,  true,  3.5263295365010903),
			new NodeData(75,150, true,  4.506106967216822)
		};
		
		NodeData comparison;
		for(int i = 0; i < expectedNodeData.length; i++) {
			comparison = tree.node_data[i];
			comparison.toString(); // Just to make sure toString() doesn't create NPE
			assertTrue(comparison.equals(expectedNodeData[i]));
		}
		
		
		Neighborhood neighb;
		boolean[] trueFalse = new boolean[]{false, true};
		for(boolean dualTree: trueFalse) {
			
			neighb= tree.query(query, 3, dualTree, true);
			
			assertTrue(MatUtils.equalsWithTolerance(expectedDists, neighb.getDistances(), 1e-8));
			assertTrue(MatUtils.equalsExactly(expectedIndices, neighb.getIndices()));
		}
	}
	
	
	@Test
	public void testQueryRadiusNoSort() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedNonSortedDists = new double[][]{
			new double[]{ 0.        ,  0.53851648,  0.50990195,  0.64807407,  0.14142136,
			        0.6164414 ,  0.51961524,  0.17320508,  0.46904158,  0.37416574,
			        0.37416574,  0.59160798,  0.54772256,  0.1       ,  0.74161985,
			        0.33166248,  0.43588989,  0.3       ,  0.64807407,  0.46904158,
			        0.59160798,  0.54772256,  0.31622777,  0.14142136,  0.14142136,
			        0.53851648,  0.53851648,  0.38729833,  0.6244998 ,  0.46904158,
			        0.37416574,  0.41231056,  0.46904158,  0.14142136,  0.17320508,
			        0.76811457,  0.45825757,  0.6164414 ,  0.59160798,  0.36055513,
			        0.58309519,  0.3       ,  0.2236068 },
			
			       new double[]{ 0.53851648,  0.        ,  0.3       ,  0.33166248,  0.60827625,
			        0.50990195,  0.42426407,  0.50990195,  0.17320508,  0.45825757,
			        0.14142136,  0.678233  ,  0.54772256,  0.70710678,  0.76157731,
			        0.78102497,  0.55677644,  0.64807407,  0.2236068 ,  0.5       ,
			        0.59160798,  0.5       ,  0.34641016,  0.24494897,  0.678233  ,
			        0.17320508,  0.3       ,  0.78740079,  0.17320508,  0.50990195,
			        0.45825757,  0.52915026,  0.54772256,  0.678233  ,  0.14142136,
			        0.36055513,  0.31622777},
			       
			       new double[]{ 0.50990195,  0.3       ,  0.        ,  0.24494897,  0.50990195,
			        0.26457513,  0.41231056,  0.43588989,  0.31622777,  0.37416574,
			        0.26457513,  0.5       ,  0.51961524,  0.75498344,  0.7       ,
			        0.50990195,  0.64807407,  0.64031242,  0.46904158,  0.50990195,
			        0.6164414 ,  0.54772256,  0.3       ,  0.33166248,  0.78102497,
			        0.31622777,  0.31622777,  0.31622777,  0.36055513,  0.48989795,
			        0.43588989,  0.3       ,  0.65574385,  0.26457513,  0.78102497,
			        0.14142136,  0.33166248},
			       
			       new double[]{ 0.64807407,  0.33166248,  0.24494897,  0.        ,  0.64807407,
			        0.33166248,  0.5       ,  0.3       ,  0.31622777,  0.37416574,
			        0.26457513,  0.51961524,  0.65574385,  0.70710678,  0.64807407,
			        0.53851648,  0.42426407,  0.54772256,  0.72111026,  0.678233  ,
			        0.17320508,  0.2236068 ,  0.31622777,  0.50990195,  0.31622777,
			        0.3       ,  0.58309519,  0.60827625,  0.3       ,  0.7       ,
			        0.26457513,  0.14142136,  0.45825757},
			       
			       new double[]{ 0.14142136,  0.60827625,  0.50990195,  0.64807407,  0.        ,
			        0.6164414 ,  0.45825757,  0.2236068 ,  0.52915026,  0.42426407,
			        0.34641016,  0.64031242,  0.54772256,  0.17320508,  0.79372539,
			        0.26457513,  0.53851648,  0.26457513,  0.56568542,  0.52915026,
			        0.57445626,  0.63245553,  0.34641016,  0.24494897,  0.28284271,
			        0.53851648,  0.57445626,  0.5       ,  0.55677644,  0.78102497,
			        0.52915026,  0.4472136 ,  0.51961524,  0.52915026,  0.24494897,
			        0.17320508,  0.72801099,  0.45825757,  0.58309519,  0.64031242,
			        0.3       ,  0.56568542,  0.33166248,  0.3       },
			       
			       new double[]{ 0.6164414 ,  0.6164414 ,  0.        ,  0.7       ,  0.34641016,
			        0.678233  ,  0.6164414 ,  0.4       ,  0.59160798,  0.33166248,
			        0.38729833,  0.53851648,  0.41231056,  0.678233  ,  0.64807407,
			        0.52915026,  0.64807407,  0.53851648,  0.45825757,  0.47958315,
			        0.60827625,  0.64807407,  0.7       ,  0.60827625,  0.37416574,
			        0.38729833,  0.36055513},
			       
			       new double[]{ 0.51961524,  0.50990195,  0.26457513,  0.33166248,  0.45825757,
			        0.        ,  0.42426407,  0.54772256,  0.47958315,  0.3       ,
			        0.48989795,  0.6164414 ,  0.50990195,  0.64807407,  0.6       ,
			        0.45825757,  0.6244998 ,  0.54772256,  0.60827625,  0.45825757,
			        0.6244998 ,  0.60827625,  0.31622777,  0.42426407,  0.47958315,
			        0.5       ,  0.47958315,  0.46904158,  0.51961524,  0.42426407,
			        0.31622777,  0.54772256,  0.4472136 ,  0.678233  ,  0.2236068 ,
			        0.77459667,  0.42426407},
			       
			       new double[]{ 0.17320508,  0.42426407,  0.41231056,  0.5       ,  0.2236068 ,
			        0.7       ,  0.42426407,  0.        ,  0.78740079,  0.33166248,
			        0.5       ,  0.2236068 ,  0.46904158,  0.7       ,  0.2       ,
			        0.42426407,  0.4472136 ,  0.37416574,  0.67082039,  0.38729833,
			        0.4472136 ,  0.41231056,  0.2236068 ,  0.2236068 ,  0.2236068 ,
			        0.37416574,  0.37416574,  0.4472136 ,  0.73484692,  0.33166248,
			        0.36055513,  0.54772256,  0.33166248,  0.74833148,  0.1       ,
			        0.24494897,  0.66332496,  0.42426407,  0.60827625,  0.46904158,
			        0.42426407,  0.45825757,  0.42426407,  0.14142136},
			       
			       new double[]{ 0.50990195,  0.43588989,  0.3       ,  0.54772256,  0.78740079,
			        0.        ,  0.55677644,  0.67082039,  0.42426407,  0.34641016,
			        0.64031242,  0.46904158,  0.48989795,  0.55677644,  0.7       ,
			        0.55677644,  0.14142136,  0.6244998 ,  0.31622777,  0.42426407,
			        0.36055513,  0.72111026},
			       
			       new double[]{ 0.46904158,  0.17320508,  0.31622777,  0.31622777,  0.52915026,
			        0.47958315,  0.33166248,  0.55677644,  0.        ,  0.78740079,
			        0.34641016,  0.17320508,  0.72801099,  0.5       ,  0.75498344,
			        0.6244998 ,  0.7       ,  0.77459667,  0.52915026,  0.51961524,
			        0.2       ,  0.4472136 ,  0.50990195,  0.4472136 ,  0.26457513,
			        0.17320508,  0.65574385,  0.        ,  0.34641016,  0.75498344,
			        0.        ,  0.55677644,  0.37416574,  0.5       ,  0.55677644,
			        0.65574385,  0.26457513,  0.74161985,  0.34641016,  0.72801099,
			        0.26457513}
		};
		
		int[][] expectedNonSortedIndices = new int[][]{
			new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 20,
			        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39,
			        40, 42, 43, 44, 45, 46, 47, 48, 49},
			
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 20, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 42, 43,
	        45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 19, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 39, 40, 42, 43, 45,
	        46, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 22, 23, 24, 25,
	        26, 27, 28, 29, 30, 34, 35, 37, 38, 39, 40, 42, 43, 45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 20,
	        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
	        39, 40, 42, 43, 44, 45, 46, 47, 48, 49},
	        
	        new int[]{ 0,  4,  5,  7, 10, 14, 15, 16, 17, 18, 19, 20, 21, 23, 26, 27, 28,
	        31, 32, 33, 36, 39, 40, 43, 44, 46, 48},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 19, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 34, 35, 37, 38, 39, 40, 42, 43, 45, 46,
	        47, 48, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 16, 17, 19, 20,
	        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38,
	        39, 40, 42, 43, 44, 45, 46, 47, 48, 49},
	        
	        new int[]{ 1,  2,  3,  6,  7,  8,  9, 11, 12, 13, 25, 29, 30, 34, 35, 37, 38,
	        41, 42, 45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 17, 19, 20, 21,
	        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40,
	        42, 43, 45, 46, 47, 48, 49}
		};
		
		
		
		Neighborhood neighb = tree.queryRadius(query, 0.8, false);
		
		// Just want to know that the total diff in matrices generated from sklearn and clust4j
		// is less than some arbitrarily low number, say one (rounding error).
		assertTrue(absDiffInMatrices(expectedNonSortedDists, neighb.getDistances()) < 1);
		
		int[][] indices = neighb.getIndices();
		for(int i = 0; i < expectedNonSortedIndices.length; i++)
			assertTrue(differenceInIdxArrays(expectedNonSortedIndices[i], indices[i]) <= 2);
	}
	
	@Test
	public void testQueryRadiusWithSort() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedSortedDists = new double[][]{
			new double[]{ 0.        ,  0.1       ,  0.14142136,  0.14142136,  0.14142136,
	        0.14142136,  0.17320508,  0.17320508,  0.2236068 ,  0.3       ,
	        0.3       ,  0.31622777,  0.33166248,  0.36055513,  0.37416574,
	        0.37416574,  0.37416574,  0.38729833,  0.41231056,  0.43588989,
	        0.45825757,  0.46904158,  0.46904158,  0.46904158,  0.46904158,
	        0.50990195,  0.51961524,  0.53851648,  0.53851648,  0.53851648,
	        0.54772256,  0.54772256,  0.58309519,  0.59160798,  0.59160798,
	        0.59160798,  0.6164414 ,  0.6164414 ,  0.6244998 ,  0.64807407,
	        0.64807407,  0.74161985,  0.76811457},
	       new double[]{ 0.        ,  0.14142136,  0.14142136,  0.17320508,  0.17320508,
	        0.17320508,  0.2236068 ,  0.24494897,  0.3       ,  0.3       ,
	        0.31622777,  0.33166248,  0.34641016,  0.36055513,  0.42426407,
	        0.45825757,  0.45825757,  0.5       ,  0.5       ,  0.50990195,
	        0.50990195,  0.50990195,  0.52915026,  0.53851648,  0.54772256,
	        0.54772256,  0.55677644,  0.59160798,  0.60827625,  0.64807407,
	        0.678233  ,  0.678233  ,  0.678233  ,  0.70710678,  0.76157731,
	        0.78102497,  0.78740079},
	       new double[]{ 0.        ,  0.14142136,  0.24494897,  0.26457513,  0.26457513,
	        0.26457513,  0.3       ,  0.3       ,  0.3       ,  0.31622777,
	        0.31622777,  0.31622777,  0.31622777,  0.33166248,  0.33166248,
	        0.36055513,  0.37416574,  0.41231056,  0.43588989,  0.43588989,
	        0.46904158,  0.48989795,  0.5       ,  0.50990195,  0.50990195,
	        0.50990195,  0.50990195,  0.51961524,  0.54772256,  0.6164414 ,
	        0.64031242,  0.64807407,  0.65574385,  0.7       ,  0.75498344,
	        0.78102497,  0.78102497},
	       new double[]{ 0.        ,  0.14142136,  0.17320508,  0.2236068 ,  0.24494897,
	        0.26457513,  0.26457513,  0.3       ,  0.3       ,  0.3       ,
	        0.31622777,  0.31622777,  0.31622777,  0.33166248,  0.33166248,
	        0.37416574,  0.42426407,  0.45825757,  0.5       ,  0.50990195,
	        0.51961524,  0.53851648,  0.54772256,  0.58309519,  0.60827625,
	        0.64807407,  0.64807407,  0.64807407,  0.65574385,  0.678233  ,
	        0.7       ,  0.70710678,  0.72111026},
	       new double[]{ 0.        ,  0.14142136,  0.17320508,  0.17320508,  0.2236068 ,
	        0.24494897,  0.24494897,  0.26457513,  0.26457513,  0.28284271,
	        0.3       ,  0.3       ,  0.33166248,  0.34641016,  0.34641016,
	        0.42426407,  0.4472136 ,  0.45825757,  0.45825757,  0.5       ,
	        0.50990195,  0.51961524,  0.52915026,  0.52915026,  0.52915026,
	        0.52915026,  0.53851648,  0.53851648,  0.54772256,  0.55677644,
	        0.56568542,  0.56568542,  0.57445626,  0.57445626,  0.58309519,
	        0.60827625,  0.6164414 ,  0.63245553,  0.64031242,  0.64031242,
	        0.64807407,  0.72801099,  0.78102497,  0.79372539},
	       new double[]{ 0.        ,  0.33166248,  0.34641016,  0.36055513,  0.37416574,
	        0.38729833,  0.38729833,  0.4       ,  0.41231056,  0.45825757,
	        0.47958315,  0.52915026,  0.53851648,  0.53851648,  0.59160798,
	        0.60827625,  0.60827625,  0.6164414 ,  0.6164414 ,  0.6164414 ,
	        0.64807407,  0.64807407,  0.64807407,  0.678233  ,  0.678233  ,
	        0.7       ,  0.7       },
	       new double[]{ 0.        ,  0.2236068 ,  0.26457513,  0.3       ,  0.31622777,
	        0.31622777,  0.33166248,  0.42426407,  0.42426407,  0.42426407,
	        0.42426407,  0.4472136 ,  0.45825757,  0.45825757,  0.45825757,
	        0.46904158,  0.47958315,  0.47958315,  0.47958315,  0.48989795,
	        0.5       ,  0.50990195,  0.50990195,  0.51961524,  0.51961524,
	        0.54772256,  0.54772256,  0.54772256,  0.6       ,  0.60827625,
	        0.60827625,  0.6164414 ,  0.6244998 ,  0.6244998 ,  0.64807407,
	        0.678233  ,  0.77459667},
	       new double[]{ 0.        ,  0.1       ,  0.14142136,  0.17320508,  0.2       ,
	        0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 ,
	        0.24494897,  0.33166248,  0.33166248,  0.33166248,  0.36055513,
	        0.37416574,  0.37416574,  0.37416574,  0.38729833,  0.41231056,
	        0.41231056,  0.42426407,  0.42426407,  0.42426407,  0.42426407,
	        0.42426407,  0.42426407,  0.4472136 ,  0.4472136 ,  0.4472136 ,
	        0.45825757,  0.46904158,  0.46904158,  0.5       ,  0.5       ,
	        0.54772256,  0.60827625,  0.66332496,  0.67082039,  0.7       ,
	        0.7       ,  0.73484692,  0.74833148,  0.78740079},
	       new double[]{ 0.        ,  0.14142136,  0.3       ,  0.31622777,  0.34641016,
	        0.36055513,  0.42426407,  0.42426407,  0.43588989,  0.46904158,
	        0.48989795,  0.50990195,  0.54772256,  0.55677644,  0.55677644,
	        0.55677644,  0.6244998 ,  0.64031242,  0.67082039,  0.7       ,
	        0.72111026,  0.78740079},
	       new double[]{ 0.        ,  0.        ,  0.        ,  0.17320508,  0.17320508,
	        0.17320508,  0.2       ,  0.26457513,  0.26457513,  0.26457513,
	        0.31622777,  0.31622777,  0.33166248,  0.34641016,  0.34641016,
	        0.34641016,  0.37416574,  0.4472136 ,  0.4472136 ,  0.46904158,
	        0.47958315,  0.5       ,  0.5       ,  0.50990195,  0.51961524,
	        0.52915026,  0.52915026,  0.55677644,  0.55677644,  0.55677644,
	        0.6244998 ,  0.65574385,  0.65574385,  0.7       ,  0.72801099,
	        0.72801099,  0.74161985,  0.75498344,  0.75498344,  0.77459667,
	        0.78740079}
		};
		
		int[][] expectedSortedIndices = new int[][]{
			new int[]{ 0, 17,  4, 39, 27, 28, 40,  7, 49, 21, 48, 26, 19, 46, 35, 11, 10,
	        31, 36, 20, 43,  9, 34, 37, 23,  2,  6, 29,  1, 30, 25, 16, 47, 24,
	        12, 45, 44,  5, 32,  3, 22, 18, 42},
	        new int[]{ 1, 45, 12, 37, 34,  9, 25, 30, 35,  2, 49,  3, 29, 47,  7, 39, 11,
	        28, 26, 38,  8,  6, 40,  0, 17, 42, 23, 27,  4, 24, 31, 43, 13, 20,
	        21, 22, 36},
	        new int[]{ 2, 47,  3, 45, 12,  6, 42, 29,  1, 35, 37, 34,  9, 49, 30, 38, 11,
	         7, 40,  8, 25, 39, 13,  0, 26,  4, 22, 17, 28, 27, 24, 23, 43, 21,
	        19, 46, 31},
	        new int[]{ 3, 47, 29, 30,  2, 12, 45, 42, 38,  8,  9, 34, 37,  6,  1, 11, 25,
	        49,  7, 35, 13, 24, 26, 39, 40,  0, 23,  4, 17, 28, 43, 22, 27},
	        new int[]{ 4,  0, 17, 40,  7, 39, 27, 19, 21, 28, 46, 49, 48, 26, 11, 10, 35,
	        43,  6, 31,  2, 36, 34, 37,  9, 23, 29, 20, 16, 32, 22, 47, 24, 30,
	        44,  1,  5, 25, 45, 12,  3, 42, 33, 18},
	        new int[]{ 5, 18, 10, 48, 44, 46, 19, 16, 21, 32, 33, 27, 31, 20, 17, 36, 43,
	         0, 15,  4, 28, 26, 39, 14, 23, 40,  7},
	        new int[]{ 6, 47,  2, 11, 42, 29,  3, 30, 49,  7, 40, 45, 22,  4, 26, 38, 37,
	         9, 34, 12, 35, 17,  1,  0, 39,  8, 24, 43, 21, 25, 28, 13, 23, 27,
	        19, 46, 48},
	        new int[]{ 7, 39, 49,  0, 17, 26, 28, 27, 11,  4, 40,  9, 34, 37, 35, 29, 30,
	        21, 23,  2, 25, 19,  1, 46, 48, 43,  6, 24, 20, 31, 47, 45, 12,  3,
	        10, 36, 44, 42, 22,  5, 16, 32, 38,  8},
	        new int[]{ 8, 38,  3, 42, 13, 47, 12, 45,  2, 29, 30,  1,  6,  9, 37, 34, 41,
	        25, 11, 35, 49,  7},
	        new int[]{34, 37,  9,  1, 30, 12, 25, 49, 29, 45,  2,  3,  7, 35, 11, 47, 39,
	        28, 26,  0,  6, 17, 40, 27, 24, 23,  4, 42, 38,  8, 20, 43, 31, 21,
	        48, 13, 46, 19, 36, 22, 10}
		};
		
		
		
		Neighborhood neighb = tree.queryRadius(query, 0.8, true);
		
		// ensure doesn't throw NPE
		assertTrue(null != neighb.toString());
		
		// ensure doesn't throw NPE
		assertTrue(null != neighb.copy());
		
		// Just want to know that the total diff in matrices generated from sklearn and clust4j
		// is less than some arbitrarily low number, say one (rounding error).
		assertTrue(absDiffInMatrices(expectedSortedDists, neighb.getDistances()) < 1);
		
		int[][] indices = neighb.getIndices();
		for(int i = 0; i < expectedSortedIndices.length; i++)
			assertTrue(differenceInIdxArrays(expectedSortedIndices[i], indices[i]) <= 2);
	}
	
	private static double absDiffInMatrices(double[][] expected, double[][] actual) {
		double sumA = 0;
		double sumB = 0;
		for(int i = 0; i < expected.length; i++) {
			sumA += VecUtils.sum(VecUtils.abs(expected[i]));
			sumB += VecUtils.sum(VecUtils.abs(actual[i]));
		}
		
		return FastMath.abs(sumA - sumB);
	}
	
	private static int differenceInIdxArrays(int[] expected, int[] actual) {
		// Check to see if the diff is <= 2
		ArrayList<Integer> aa = new ArrayList<Integer>();
		ArrayList<Integer> bb = new ArrayList<Integer>();
		
		for(int in: expected)
			aa.add(in);
		for(int in: actual)
			bb.add(in);
		
		ArrayList<Integer> larger = aa.size() > bb.size() ? aa : bb;
		ArrayList<Integer> smaller= aa.equals(larger) ? bb : aa;
		larger.removeAll(smaller);
		
		return larger.size();
	}
	
	private void addOne(MutableDouble d) {
		d.value++;
	}
	
	@Test
	public void testMutDouble2() {
		MutableDouble d= new MutableDouble();
		addOne(d);
		assertTrue(d.value == 1);
	}
	
	@Test
	public void testTwoPointCorrelation() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		int[] corSingle, corDual;
		corSingle = tree.twoPointCorrelation(query, 2.5, false);
		corDual = tree.twoPointCorrelation(query, 2.5, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(542, 10)));
		
		corSingle = tree.twoPointCorrelation(query, 1.5, false);
		corDual = tree.twoPointCorrelation(query, 1.5, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(489, 10)));
		
		corSingle = tree.twoPointCorrelation(query, 25, false);
		corDual = tree.twoPointCorrelation(query, 25, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(1500, 10)));
		
		

		corSingle = tree.twoPointCorrelation(query, 0, false);
		corDual = tree.twoPointCorrelation(query, 0, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(12, 10)));
		
		corSingle = tree.twoPointCorrelation(query, -1, false);
		corDual = tree.twoPointCorrelation(query, -1, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(0, 10)));
		
		// Test a big query now, just to ensure no exceptions are thrown...
		final double[][] X = IRIS.getData();
		tree.twoPointCorrelation(X, -1, false);
		tree.twoPointCorrelation(X, -1, true);
		tree.twoPointCorrelation(X, -1.0);
		tree.twoPointCorrelation(X, new double[]{1,2});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testTwoPointCorrelationExcept1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.twoPointCorrelation(new double[][]{new double[]{1,2}}, new double[]{1.5});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testTwoPointCorrelationExcept2() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.twoPointCorrelation(new double[][]{new double[]{1,2}}, 1.5);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestDimException() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{1,2}}, 150.0, true);
	}
	
	@Test
	public void radiusQueryTestAllInRadius() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5,2.5}}, 150.0, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestMPrimeDimMismatch1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5,2.5}}, 
			new double[]{1,2,3,4,5}, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestNDimMismatch2() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5}}, 
			new double[]{5}, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void queryNDimMismatch1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.query(new double[][]{new double[]{1,2}}, 2, true, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testKernelDimMismatch() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.kernelDensity(new double[][]{new double[]{1.0}}, 1.0, PartialKernelDensity.LOG_COSINE);
	}
	
	@Test
	public void testNodeDataEquals() {
		NodeData n1 = new NodeData();
		NodeData n2 = new NodeData(1,2,true,1.9);
		
		assertTrue(n1.equals(n1));
		assertFalse(n1.equals(n2));
		assertFalse(n1.equals(new String()));
	}
	
	@Test
	public void testInfDist() {
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		KDTree k = new KDTree(mat, Distance.CHEBYSHEV);
		Neighborhood n = k.query(mat);
		Neighborhood p = k.query(mat, 1, false, true);
		assertTrue(n.equals(p));
		assertTrue(n.equals(n));
		assertFalse(n.equals("asdf"));
		
		Neighborhood res = new Neighborhood(
			new double[][]{
				new double[]{0.0},
				new double[]{0.0},
				new double[]{0.0}
			},	
			
			new int[][]{
				new int[]{0},
				new int[]{1},
				new int[]{2}
			}
		);
		
		assertTrue(n.equals(res));
		final int[] corr = k.twoPointCorrelation(mat.getDataRef(), new double[]{1,2,3});
		assertTrue(VecUtils.equalsExactly(corr, new int[]{3,3,7}));
		assertTrue(k.infinity_dist);
	}
	
	@Test
	public void testWarn() {
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		KDTree k = new KDTree(mat, new HaversineDistance(), new KMeans(mat,1));
		assertTrue(k.logger.hasWarnings());
	}
	
	@Test
	public void testImmutability() {
		double[][] a = MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3);
		
		double[][] b = MatUtils.copy(a);
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		
		KDTree k = new KDTree(mat, Distance.EUCLIDEAN);
		k.query(a);
		
		assertTrue(MatUtils.equalsExactly(b, a)); // assert immutability
	}
}
