package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

public class ClustStructTests {

	@Test
	public void test1() {
		Cluster a = new Cluster();
		a.add(new double[]{1d,2d,3d,4d});
		
		Cluster b = new Cluster();
		b.add(new double[]{1d,2d,3d,4d});
		
		assertFalse(a.equals(b));
	}
	
	@Test
	public void test2() {
		Cluster a = new Cluster();
		a.add(new double[]{1d,2d,3d,4d});
		a.add(new double[]{3d,2d,6d,9d});
		
		assertFalse(a.centroid().equals(new double[]{2d, 2d, 4.5, 6.5}));
	}

}
