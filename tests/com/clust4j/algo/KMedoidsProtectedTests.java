package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

public class KMedoidsProtectedTests {
	/**
	 * This is the method as it is used in the KMedoids class,
	 * except that the distance matrix is passed in
	 * @param indices
	 * @param med_idx
	 * @return
	 */
	protected static double getCost(ArrayList<Integer> indices, final int med_idx, final double[][] dist_mat) {
		double cost = 0;
		for(Integer idx: indices)
			cost += dist_mat[FastMath.min(idx, med_idx)][FastMath.max(idx, med_idx)];
		return cost;
	}
	
	
	@Test
	public void test() {
		final double[][] distanceMatrix = new double[][] {
			new double[]{0,1,2,3},
			new double[]{0,0,1,2},
			new double[]{0,0,0,1},
			new double[]{0,0,0,0}
		};
		
		final int med_idx = 2;
		
		final ArrayList<Integer> belonging = new ArrayList<Integer>();
		belonging.add(0); belonging.add(1); belonging.add(2); belonging.add(3);
		assertTrue(getCost(belonging, med_idx, distanceMatrix) == 4);
	}

}
