package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

/**
 * Most heap tests happen in the HDBSCAN tests, but for
 * the sake of consistent structure, here's one or two tests
 * @author Taylor G Smith
 */
public class HeapTests {

	@Test
	public void test() {
		SimpleHeap<Integer> s = new SimpleHeap<Integer>();
		s.add(1);
		assertTrue(s.pop() == 1);
		assertTrue(s.size() == 0);
	}

}
