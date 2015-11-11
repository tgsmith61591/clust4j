package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.Iterator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeMap;

import org.junit.Test;

public class TestMapUtils {

	@Test
	public void testSorted1() {
		TreeMap<Integer, Double> t1 = new TreeMap<Integer, Double>();
		
		t1.put(1, 80d);
		t1.put(2, 45d);
		t1.put(3, 1d);
		
		SortedSet<Map.Entry<Integer,Double>> sorted = ClustUtils.sortEntriesByValue(t1);
		Iterator<Map.Entry<Integer, Double>> iter = sorted.iterator();
		
		assertTrue(iter.next().getKey().equals(3));
		assertTrue(iter.next().getKey().equals(2));
		assertTrue(iter.next().getKey().equals(1));
	}
	
	@Test
	public void testSortedDesc1() {
		TreeMap<Integer, Double> t1 = new TreeMap<Integer, Double>();
		
		t1.put(1, 80d);
		t1.put(2, 45d);
		t1.put(3, 1d);
		
		SortedSet<Map.Entry<Integer,Double>> sorted = ClustUtils.sortEntriesByValue(t1, true);
		Iterator<Map.Entry<Integer, Double>> iter = sorted.iterator();
		
		assertTrue(iter.next().getKey().equals(1));
		assertTrue(iter.next().getKey().equals(2));
		assertTrue(iter.next().getKey().equals(3));
	}

	@Test
	public void testSortedDesc2() {
		TreeMap<Integer, Double> t1 = new TreeMap<Integer, Double>();
		
		t1.put(90, 80d);
		t1.put(45, 45d);
		t1.put(20, 1d);
		
		SortedSet<Map.Entry<Integer,Double>> sorted = ClustUtils.sortEntriesByValue(t1, true);
		Iterator<Map.Entry<Integer, Double>> iter = sorted.iterator();
		
		assertTrue(iter.next().getKey().equals(90));
		assertTrue(iter.next().getKey().equals(45));
		assertTrue(iter.next().getKey().equals(20));
	}
}
