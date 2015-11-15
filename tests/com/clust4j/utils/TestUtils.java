package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeMap;

import org.junit.Test;

import com.clust4j.utils.ClustUtils.SortedHashableIntSet;

public class TestUtils {

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
	
	@Test
	public void testSortedIntSet() {
		SortedHashableIntSet a = new SortedHashableIntSet();
		SortedHashableIntSet b = new SortedHashableIntSet();
		
		a.add(1); a.add(2); a.add(3); a.add(4); a.add(5);
		b.add(5); b.add(4); b.add(3); b.add(2); b.add(1);
		
		assertTrue(a.equals(b));
		b.add(6);
		assertFalse(a.equals(b));
		a.add(7);
		assertFalse(a.equals(b));
		a.add(6); b.add(7);
		assertTrue(a.equals(b));
		
		HashSet<SortedHashableIntSet> sets = new HashSet<SortedHashableIntSet>();
		sets.add(a);
		assertTrue(sets.contains(b));
	}
}
