package com.clust4j.algo;

import java.io.IOException;

import org.junit.Test;

public interface ClusterTest {
	@Test public void testDefConst();
	@Test public void testArgConst();
	@Test public void testPlannerConst();
	@Test public void testFit();
	@Test public void testFromPlanner();
	@Test public void testSerialization() throws IOException, ClassNotFoundException;
}
