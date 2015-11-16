package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import org.junit.Test;

public class BinarySearchTreeTests {
	private int big_size = 1_000;
	private boolean slow_test = true;

	@Test
	public void bstTestA() {
		BinarySearchTree<Integer> b = new BinarySearchTree<>(1);
		
		assertTrue(b.size() == 1);
		assertTrue(b.getRoot().getValue() == 1);
		
		b.add(2);
		assertTrue(b.size() == 2);
		assertTrue(b.values().size() == 2);
		assertTrue(b.getRoot().leftChild().size() == 1);
		
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
		
		assertTrue(b.getRoot().getValue() == 2);
		assertTrue(b.getRoot().leftChild().getValue() == 1);
		assertTrue(b.getRoot().leftChild().leftChild().getValue() == 0);
		assertTrue(!b.getRoot().leftChild().leftChild().hasLeft());
		
		assertTrue(b.getRoot().rightChild().getValue() == 5);
		assertTrue(b.getRoot().rightChild().leftChild().getValue() == 3);
		
		assertTrue(b.remove(3));
		assertTrue(b.size() == 4);
		
		assertFalse(b.remove(9));
		assertTrue(b.size() == 4);
		
		System.out.println(b.getRoot());
	}
	
	@Test
	public void bstTestB() {
		BinarySearchTree<Integer> b = new BinarySearchTree<>();
		assertFalse(b.remove(8));
	}
	
	@Test
	public void bstTestIterators() {
		BinarySearchTree<Integer> b = new BinarySearchTree<>(1);
		b.add(2);
		b.add(0);
		b.add(3);
		b.add(5);
		
		/*
		        2
		      /   \
		     1     5
		    /     /
		   0     3
		*/
		
		int i = 0;
		Integer[] array = new Integer[b.size()];
		
		
		// IN ORDER TESTS
		Iterator<Integer> inOrder = b.inOrderIterator();
		while(inOrder.hasNext())
			array[i++] = inOrder.next();
		
		assertTrue(array[0].intValue() == 0);
		assertTrue(array[1].intValue() == 1);
		assertTrue(array[2].intValue() == 2);
		assertTrue(array[3].intValue() == 3);
		assertTrue(array[4].intValue() == 5);
		
		// POST ORDER TESTS
		i = 0; 
		array = new Integer[b.size()];
		Iterator<Integer> postOrder = b.postOrderIterator();
		while(postOrder.hasNext())
			array[i++] = postOrder.next();
		
		assertTrue(array[0].intValue() == 0);
		assertTrue(array[1].intValue() == 1);
		assertTrue(array[2].intValue() == 3);
		assertTrue(array[3].intValue() == 5);
		assertTrue(array[4].intValue() == 2);
		
		// PRE ORDER TESTS
		i = 0; 
		array = new Integer[b.size()];
		Iterator<Integer> preOrder = b.preOrderIterator();
		while(preOrder.hasNext())
			array[i++] = preOrder.next();
		
		assertTrue(array[0].intValue() == 2);
		assertTrue(array[1].intValue() == 1);
		assertTrue(array[2].intValue() == 0);
		assertTrue(array[3].intValue() == 5);
		assertTrue(array[4].intValue() == 3);
		
		i = 0;
		for(Integer integer: b)
			assertTrue(array[i++] == integer.intValue());
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
			System.out.println("Faster test completed in "+ (double)faster/1000+" secs");
			
			
			
			
			start = System.currentTimeMillis();
			bst = new BinarySearchTree<>();
			
			i = 0;
			while(i++ < big_size)
				bst.add(rand.nextDouble());
	
			assertTrue(bst.size() == big_size);
			final long slower = System.currentTimeMillis() - start;
			System.out.println("Slower test completed in "+ (double)slower/1000+" secs");
			
			
			
			assertTrue(faster < slower);
		}
		
		assertTrue(true);
	}
}
