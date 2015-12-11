package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.Test;
import com.clust4j.algo.HierarchicalAgglomerativeClusterer.HeapUtils;

public class HeapTesting {
	
	@Test
	public void testHeapifier() {
		// Test heapify initial
		final ArrayList<Integer> x = new ArrayList<>(Arrays.asList(new Integer[]{19, 56, 1, 52, 7, 2, 23}));
		HeapUtils.heapifyInPlace(x);
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{1, 7, 2, 52, 56, 19, 23}))));
		
		// Test push pop
		Integer i = HeapUtils.heapPushPop(x, 2);
		assertTrue(i.equals(1));
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 2, 52, 56, 19, 23}))));
		
		// Test pop
		i = HeapUtils.heapPop(x);
		assertTrue(i.equals(2));
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 19, 52, 56, 23}))));
		
		// Test push
		HeapUtils.heapPush(x, 9);
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 9, 52, 56, 23, 19}))));
	}

}
