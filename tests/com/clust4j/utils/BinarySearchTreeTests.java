package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.junit.Test;

public class BinarySearchTreeTests {
	private int big_size = 10_000;
	private boolean slow_test = true;

	@Test
	public void bstTestA() {
		BinarySearchTree<Integer> b = new BinarySearchTree<>(1);
		
		assertTrue(b.size() == 1);
		assertTrue(b.root().getValue() == 1);
		
		b.add(2);
		assertTrue(b.size() == 2);
		assertTrue(b.values().size() == 2);
		
		b.add(0);
		b.add(3);
		
		assertTrue(b.size() == 4);
		assertTrue(b.values().size() == 4);
		b.add(5);
		
		/* If unbalanced, would look like this:
		 
		      1
		     / \
		    0   2
		         \ 
		          3
		           \
		            5
		*/
		
		// But we are asserting that it is now balanced... should look like this:
		/*
		 
		         2
		       /   \
		      1     5
		     /     /
		    0     3
		 */
		
		assertTrue(b.root().getValue() == 2);
		assertTrue(b.root().leftChild().getValue() == 1);
		assertTrue(b.root().leftChild().leftChild().getValue() == 0);
		assertTrue(!b.root().leftChild().leftChild().hasLeft());
		
		assertTrue(b.root().rightChild().getValue() == 5);
		assertTrue(b.root().rightChild().leftChild().getValue() == 3);
		
		assertTrue(b.remove(3));
		assertTrue(b.size() == 4);
		
		assertFalse(b.remove(9));
		assertTrue(b.size() == 4);
	}
	
	@Test
	public void bstTestB() {
		BinarySearchTree<Integer> b = new BinarySearchTree<>();
		assertFalse(b.remove(8));
	}
	
	@Test
	public void bstTestHuge() {
		if(slow_test) {
			long start = System.currentTimeMillis();
			BinarySearchTree<Double> bst = new BinarySearchTree<>();
			final ArrayList<Double> vals = new ArrayList<Double>();
			final Random rand = new Random();
			
			int i = 0;
			while(i++ < big_size)
				vals.add(rand.nextDouble());
	
			// Big batch add
			bst.addAll(vals);
			final int size = bst.size();
			assertTrue(size == big_size);
			final long faster = System.currentTimeMillis() - start;
			System.out.println("Faster test completed in "+ faster/1000+" secs");
			
			
			
			
			start = System.currentTimeMillis();
			bst = new BinarySearchTree<>();
			
			i = 0;
			while(i++ < big_size)
				bst.add(rand.nextDouble());
	
			assertTrue(bst.size() == big_size);
			final long slower = System.currentTimeMillis() - start;
			System.out.println("Slower test completed in "+ slower/1000+" secs");
			
			
			
			assertTrue(faster < slower);
		}
		
		assertTrue(true);
	}
}
