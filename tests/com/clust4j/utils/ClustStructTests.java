package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

public class ClustStructTests {

	@Test
	public void test1() {
		AgglomCluster a = new AgglomCluster();
		a.add(new double[]{1d,2d,3d,4d});
		
		AgglomCluster b = new AgglomCluster();
		b.add(new double[]{1d,2d,3d,4d});
		
		assertFalse(a.equals(b));
	}
	
	@Test
	public void test2() {
		AgglomCluster a = new AgglomCluster();
		a.add(new double[]{1d,2d,3d,4d});
		a.add(new double[]{3d,2d,6d,9d});
		
		assertFalse(a.centroid().equals(new double[]{2d, 2d, 4.5, 6.5}));
	}

}
