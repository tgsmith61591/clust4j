package com.clust4j.algo;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.HierarchicalAgglomerative.Linkage;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.MatrixFormatter;

public class HierTests {
	private static Array2DRowRealMatrix matrix = ClustTests.getRandom(250, 10);
	static final MatrixFormatter formatter = new MatrixFormatter();
	
	@Test
	public void testCondensedIdx() {
		assertTrue(ClustUtils.getIndexFromFlattenedVec(10, 3, 4) == 24);
	}
	
	@Test
	public void testCut() {
		final double[][] children = new double[][] {
			new double[]{0,2},
			new double[]{1,3}
		};
		
		final int n_clusters = 2, n_leaves = 3;
		int[] l = HierarchicalAgglomerative.hcCut(n_clusters, children, n_leaves);
		assertTrue(l[0]==0 && l[1]==1 && l[2]==0);
	}
	
	@Test
	public void testRandom() {
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(matrix,
				new HierarchicalAgglomerative
					.HierarchicalPlanner().setVerbose(false));
		hac.fit();
	}

	@Test
	public void testMore() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		int[] labels;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		for(Linkage linkage: HierarchicalAgglomerative.Linkage.values()) {
			HierarchicalAgglomerative hac = 
				new HierarchicalAgglomerative(mat,
					new HierarchicalAgglomerative
						.HierarchicalPlanner()
							.setLinkage(linkage)
							.setVerbose(false)).fit();
			
			labels = hac.getLabels();
			assertTrue(labels[0] == labels[2]);
		}
	}
	
	@Test
	public void testKernel() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		int[] labels;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		for(Linkage linkage: HierarchicalAgglomerative.Linkage.values()) {
			HierarchicalAgglomerative hac = 
				new HierarchicalAgglomerative(mat,
					new HierarchicalAgglomerative
						.HierarchicalPlanner()
							.setLinkage(linkage)
							.setSep(new GaussianKernel())
							.setVerbose(false)).fit();
			
			labels = hac.getLabels();
			assertTrue(labels[0] == labels[2]);
		}
	}
	
	@Test
	public void testTreeLinkage() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(mat,
				new HierarchicalAgglomerative
					.HierarchicalPlanner()
						.setLinkage(Linkage.AVERAGE)
						.setVerbose(false));
		
		//final double[][] dist_mat = com.clust4j.utils.ClustUtils.distanceUpperTriangMatrix(mat, hac.getSeparabilityMetric());
		//System.out.println(formatter.format(new Array2DRowRealMatrix(dist_mat,false)));
		
		HierarchicalAgglomerative.AverageLinkageTree a = hac.new AverageLinkageTree();
		final double[][] Z = a.linkage();
		assertTrue(Z[0][0] == 0 && Z[0][1] == 2 && Z[1][0] == 1 && Z[1][1]==3);
	}
	
	@Test
	public void loadTest() {
		Array2DRowRealMatrix mat = ClustTests.getRandom(2500, 10);
		new HierarchicalAgglomerative(mat,
			new HierarchicalAgglomerative
				.HierarchicalPlanner()
					.setLinkage(Linkage.AVERAGE)
					.setVerbose(true)).fit().getLabels();
	}
	
	//@Test // -- takes way too long..
	public void loadTest2() {
		Array2DRowRealMatrix mat = ClustTests.getRandom(50000, 10);
		boolean exception = false;
		
		try {
			new HierarchicalAgglomerative(mat,
				new HierarchicalAgglomerative
					.HierarchicalPlanner()
						.setLinkage(Linkage.AVERAGE)
						.setVerbose(true)).fit().getLabels();
		} catch(OutOfMemoryError | StackOverflowError e) {
			exception = true;
		}
		
		assertTrue(exception);
	}
	
	@Test
	public void loadTestKernel() {
		Array2DRowRealMatrix mat = ClustTests.getRandom(2500, 10);
		new HierarchicalAgglomerative(mat,
			new HierarchicalAgglomerative
				.HierarchicalPlanner()
					.setLinkage(Linkage.AVERAGE)
					.setSep(new GaussianKernel())
					.setVerbose(true)).fit().getLabels();
	}
}
