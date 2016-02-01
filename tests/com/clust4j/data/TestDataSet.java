package com.clust4j.data;

import static org.junit.Assert.*;

import java.text.DecimalFormat;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.AffinityPropagation;
import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.HierarchicalAgglomerative;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.MeanShift;
import com.clust4j.utils.Classifier;

public class TestDataSet {
	private final static DecimalFormat df = new DecimalFormat("##.##");

	@Test
	public void testIris() {
		DataSet iris = ExampleDataSets.IRIS;
		final int len = iris.getDataRef().getRowDimension();
		DataSet shuffled = iris.shuffle();
		assertTrue(shuffled.getDataRef().getRowDimension() == len);
		
		// Test that no reference carried over...
		shuffled.getHeaderRef()[0] = "TESTING!";
		assertTrue( !iris.getHeaderRef()[0].equals(shuffled.getHeaderRef()[0]) );
	}
	
	private static String formatPct(double num) {
		return df.format(num * 100) + "%";
	}
	
	private static void stdout(Classifier model, int[] actual, boolean b) {
		String nm = ((AbstractClusterer)model).getName();
		System.out.println(nm+" (scale = "+b+"):  " + formatPct(model.score(actual)) );
	}

	private void testAlgos(boolean shuffle) {
		System.out.println(" ========== " + "Testing with shuffle" + 
				(shuffle ? " enabled" : " disabled") + 
				" ========== ");
		
		DataSet iris = ExampleDataSets.IRIS;
		DataSet shuffled = shuffle ? iris.shuffle() : iris;
		
		final Array2DRowRealMatrix data = shuffled.getDataRef();
		final int[] actual = shuffled.getLabels();
		
		final boolean verbose = false;
		final boolean[] scale = new boolean[]{false, true};
		for(boolean b: scale) {
			
			// Go down the line alpha...
			AffinityPropagation ap = new AffinityPropagation(data, 
				new AffinityPropagation.AffinityPropagationPlanner()
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(ap, actual, b);
			
			
			DBSCAN db = new DBSCAN(data, 
				new DBSCAN.DBSCANPlanner()
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(db, actual, b);
			
			/* still in development...
			HDBSCAN hdb = new HDBSCAN(data, 
				new HDBSCAN.HDBSCANPlanner()
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(hdb, actual, b);
			*/
			
			HierarchicalAgglomerative ha = new HierarchicalAgglomerative(data, 
				new HierarchicalAgglomerative.HierarchicalPlanner()
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(ha, actual, b);
			
			
			KMeans kmn = new KMeans(data, 
				new KMeans.KMeansPlanner(3)
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(kmn, actual, b);
			
			KMedoids kmd = new KMedoids(data, 
				new KMedoids.KMedoidsPlanner(3)
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(kmd, actual, b);
			
			MeanShift ms = new MeanShift(data, 
				new MeanShift.MeanShiftPlanner()
					.setScale(b)
					.setVerbose(verbose)).fit();
			stdout(ms, actual, b);
			
			
			
			System.out.println();
		}
		
		System.out.println();
	}
	
	@Test
	public void testDifferentAlgorithmWithShuffle() {
		testAlgos(true);
	}
	
	@Test
	public void testDifferentAlgorithmNoShuffle() {
		testAlgos(false);
	}
}
