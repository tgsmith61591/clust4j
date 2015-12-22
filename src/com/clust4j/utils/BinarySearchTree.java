package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 * An implementation of a the classic self-balancing Binary Search Tree.
 * All generics must implement Comparable, as they are sorted to maintain
 * structure. The class provides a default iterator which navigates the tree
 * in pre-order, but provides methods to navigate in both POST or IN order
 * in addition.
 * 
 * <p>
 * This tree implements Collection, and as a general rule of thumb any batch
 * adds or removals will be faster for large data, as they require rebalancing.
 * See {@link #addAll(Collection)}, {@link #removeAll(Collection)} and
 * {@link #retainAll(Collection)}.
 * 
 * @author Taylor G Smith
 *
 * @param <T>
 */
public class BinarySearchTree<T extends Comparable<? super T>> 
		extends AbstractBinaryTree<T>
		implements java.io.Serializable, Iterable<T>, Collection<T> {
	private static final long serialVersionUID = -478846685921837986L;
	private int size = -1; // Used to cache size to avoid many searches...
	
	/**
	 * Keeps track of structural modifications in order to
	 * avoid ConcurrentModificationExceptions from the iterator
	 */
	private int modCount = 0;
	
	
	
	/**
	 * Generate a new, default comparator
	 */
	private final static <T extends Comparable<? super T>> Comparator<T> defaultComparator() {
		return new Comparator<T>() {
			@Override public int compare(T o1, T o2) {
				return o1.compareTo(o2);
			}
		};
	}
	
	/**
	 * The comparator used for sorting, assigning left/right
	 */
	transient private Comparator<T> comparator;

	/**
	 * Stores the current root of the BST
	 */
	private BSTNode<T> root = null;
	
	
	
	
	/* ---------------------- CONSTRUCTORS ---------------------- */
	@SuppressWarnings("unchecked") public BinarySearchTree() {
		this((Comparator<T>) defaultComparator());
	}
	
	public BinarySearchTree(final Comparator<T> comparator) {
		super();
		this.comparator = comparator;
	}
	
	@SuppressWarnings("unchecked") public BinarySearchTree(T root) {
		this((Comparator<T>) defaultComparator(), root);
	}
	
	public BinarySearchTree(final Comparator<T> comparator, final T root) {
		super();
		this.root = new BSTNode<>(root, comparator);
		this.comparator = comparator;
	}
	
	@SuppressWarnings("unchecked") public BinarySearchTree(final Collection<T> coll) {
		this((Comparator<T>) defaultComparator(), coll);
	}
	
	public BinarySearchTree(final Comparator<T> comparator, final Collection<T> coll) {
		super();
		this.comparator = comparator;
		addAll(coll);
	}
	
	
	

	
	/* ---------------------- METHODS ---------------------- */
	@Override
	public boolean add(T t) {
		if(null == root)
			root = new BSTNode<>(t, comparator);
		else root.add(t);
		
		balance(); // Increments modCount!
		return true;
	}
	
	@SuppressWarnings("unchecked") 
	@Override
	public boolean addAll(Collection<? extends T> values) {
		List<T> vals;
		if(values instanceof List)
			vals = (List<T>)values;
		else vals = new ArrayList<T>(values);
		
		if(root != null)
			vals.addAll(root.values());
		
		balance(vals); // Increments modCount!
		return true;
	}
	
	private void balance() {
		ArrayList<T> values = (ArrayList<T>) values();
		balance(values); // Increments modCount!
	}
	
	/**
	 * Rebalances the tree after a removal operation
	 * or after a batch add operation
	 * @param values
	 */
	private void balance(final List<T> values) {
		Collections.sort(values, comparator);
		
		// Re-insert
		root = null;
		size = values.size();
		modCount++;
		balanceRecursive(0, size, values);
	}
	
	private void balanceRecursive(int low, int high, List<T> coll){
	    if(low == high)
	        return;

	    int midpoint = (low + high)/2;
	    T insert = coll.get(midpoint);
	    
	    if(null == root)
	    	root = new BSTNode<>(insert, comparator);
	    else root.add(insert);

	    balanceRecursive(midpoint+1, high, coll);
	    balanceRecursive(low, midpoint, coll);  
	}
	
	@Override
	public BSTNode<T> getRoot() {
		return root;
	}


	@Override
	public boolean isEmpty() {
		return size <= 0; // It can be negative one too if cleared...
	}
	
	/**
	 * By default returns a BSTPreOrderIterator
	 */
	@Override
	public Iterator<T> iterator() {
		return preOrderIterator();
	}

	@Override
	public boolean contains(Object o) {
		// Could be faster by using locate in the root, but we
		// cannot be certain that 'o' is comparable...
		return null == root ? false : values().contains(o);
	}

	@Override
	public Object[] toArray() {
		return null == root ? new Object[0] : values().toArray();
	}

	/**
	 * Returns the data in pre-order
	 * @param a
	 * @return
	 */
	@SuppressWarnings("hiding")
	@Override
	public <T> T[] toArray(T[] a) {
		if(null == root)
			return a;
		
		@SuppressWarnings("unchecked") 
		final ArrayList<T> data = (ArrayList<T>) values();
		
		if(a.length != data.size())
			throw new DimensionMismatchException(a.length, data.size());
		
		for(int i = 0; i < a.length; i++)
			a[i] = data.get(i);
		
		return a;
	}

	@Override
	public boolean containsAll(Collection<?> c) {
		return null == root ? false : values().containsAll(c);
	}

	@Override
	public void clear() {
		root = null;
		modCount++;
		size = -1;
	}
	
	public BSTNode<T> locate(T value) {
		BaseTreeNode<T> b = super.locate(value);
		return null == b ? null : (BSTNode<T>) b;
	}
	
	public Iterator<T> inOrderIterator() {
		return new BSTInOrderIterator(root);
	}
	
	public Iterator<T> postOrderIterator() {
		return new BSTPostOrderIterator(root);
	}
	
	public Iterator<T> preOrderIterator() {
		return new BSTPreOrderIterator(root);
	}
	
	@Override
	public boolean prune(final BaseBinaryTreeNode<T> node) {
		if(!(node instanceof BSTNode))
			throw new IllegalArgumentException("illegal node type");
		
		final boolean b;
		if(b = super.prune(node))
			balance(); // Increments modCount!
		
		return b;
	}
	
	/**
	 * Removes the first node whose value's compareTo()
	 * method equals zero with the provided value.
	 * @param value
	 * @return true if found and removed from tree
	 */
	@Override
	public boolean remove(Object value) {
		if(null == root)
			return false;
		
		ArrayList<T> values = (ArrayList<T>) values();
		final boolean b = values.remove(value);
		
		if(b)
			balance(values); // Increments modCount!
		return b;
	}

	@Override
	public boolean retainAll(Collection<?> c) {
		if(null == root)
			return false;
		
		final ArrayList<T> values = (ArrayList<T>) values();
		final boolean b = values.retainAll(c);
		
		if(b)
			balance(values); // Modifies modCount!
		
		return b;
	}
	
	@Override
	public boolean removeAll(Collection<?> remove) {
		if(null == root)
			return false;
		
		ArrayList<T> values = (ArrayList<T>) root.values();
		final boolean b = values.removeAll(remove);
		
		if(b)
			balance(values); // Increments modCount!
		
		return b;
	}
	
	@Override
	public int size() {
		return null == getRoot() ? 0 : 
			size == -1 ? size = getRoot().size() : 
				size;
	}
	
	@Override
	public Collection<T> values() {
		return null == root ? null : root.values();
	}
	
	
	
	
	
	
	/**
	 * Implements nearly all tree traversal logic. Since it 
	 * contains very sensitive references to potentially mutable
	 * datastructures, the decision was made to hide visibility
	 * to all but getter methods in this innerclass, and only allow access 
	 * to the values themselves. This is especially important when considered that a 
	 * tree balancing operation will likely reassign the root as a new
	 * BSTNode by value, likely destroying the integrety to a stored
	 * reference to BSTNode. This could cause unforeseen NullPointerExceptions
	 * on the client side.
	 * 
	 * @author Taylor G Smith
	 * 
	 * @param <T>
	 */
	final public static class BSTNode<T extends Comparable<? super T>> 
			extends AbstractBinaryTree.BaseBinaryTreeNode<T>
			implements java.io.Serializable {
		private static final long serialVersionUID = -4243106326564743392L;
		
		private T value;
		private BSTNode<T> right = null;
		private BSTNode<T> left = null;
		transient private Comparator<T> comparator;
		
		private BSTNode(T t, final Comparator<T> comparator) {
			super();
			this.value = t;
			this.comparator = comparator;
		}
		
		protected void add(T t) {
			addRecurse(this, t);
		}
		
		private static <T extends Comparable<? super T>> boolean addRecurse(BSTNode<T> root, T value) {
			final boolean left = root.goesLeft(value);
			if(left && !root.hasLeft()) {
				root.left = new BSTNode<>(value, root.comparator);
				return left;
			} else if(!left && !root.hasRight()) {
				root.right = new BSTNode<>(value, root.comparator);
				return left;
			}
			
			return addRecurse(left ? root.left : root.right, value);
		}
		
		@Override
		public T getValue() {
			return value;
		}
		
		private boolean goesLeft(T incomingValue) {
			return comparator.compare(incomingValue, value) < 0;
		}
		
		@Override
		public boolean hasLeft() {
			return left != null;
		}
		
		@Override
		public boolean hasRight() {
			return right != null;
		}
		
		@Override
		public BSTNode<T> leftChild() {
			return left;
		}
		
		@Override
		protected BSTNode<T> locate(T value) {
			final int comparison = comparator.compare(this.value, value);
			if(comparison == 0) // this is the first node to have the value
				return this;
			if(comparison == -1)
				return hasLeft() ? left.locate(value) : null;
			return hasRight() ? right.locate(value) : null;
		}
		
		/**
		 * Prunes all children from this node
		 */
		@Override
		protected void prune() {
			left = null;
			right = null;
		}
		
		@Override
		public BSTNode<T> rightChild() {
			return right;
		}
		
		/**
		 * Collects the values of the tree in PRE ORDER
		 * @return
		 */
		@Override
		public Collection<T> values() {
			final ArrayList<T> values = new ArrayList<T>();
			return valuesRecurse(this, values);
		}
		
		private final static <T extends Comparable<? super T>> Collection<T> valuesRecurse(
				final BSTNode<T> root, 
				final Collection<T> coll) 
		{
			coll.add(root.value);
			if(root.hasLeft())
				valuesRecurse(root.left, coll);
			if(root.hasRight())
				valuesRecurse(root.right, coll);
			return coll;
		}
		
	}// End BSTNode
	
	
	
	
	private abstract class BSTIterator implements Iterator<T> {
		private final ArrayList<T> internal;
		private int cursor;
		private int lastRet = -1;
		int expectedModCount = modCount;
		
		public BSTIterator(BSTNode<T> theRoot) {
			internal = null == theRoot ? new ArrayList<T>() : 
				getOrderedValues(theRoot);
		}

		@Override
		public boolean hasNext() {
			return cursor != size;
		}

		@Override
		public T next() {
			checkForComodification();
			
			int i = cursor;
			if(i >= size) // If .next() called and does not have next
				throw new NoSuchElementException();
			
			cursor = i + 1;
			return internal.get(lastRet = i);
		}

		/**
		 * Removes the value from the BinarySearchTree, and requires a 
		 * full rebalancing operation. Can be expensive in large trees.
		 * If an item is removed, the ordering of the nodes WILL NOT CHANGE
		 * in this iterator until a new one is initialized.
		 */
		@Override
		public void remove() {
			if(lastRet < 0)
				throw new IllegalStateException();
			checkForComodification();
			
			try {
				// Remove while avoiding ConcurrentModificationException
				BinarySearchTree.this.remove(internal.get(lastRet));
				cursor = lastRet;
				lastRet = -1; // To avoid two concurrent removes without a next()
				expectedModCount = modCount;
			} catch(IndexOutOfBoundsException e) { // Incredibly rare corner case...
				throw new ConcurrentModificationException();
			}
		}
		
		final void checkForComodification() {
			if(modCount != expectedModCount)
				throw new ConcurrentModificationException();
		}
		
		/**
		 * Each extending iterator should implement this method
		 * in correspondence to the required ordering
		 * @param theRoot
		 * @return
		 */
		protected abstract ArrayList<T> getOrderedValues(BSTNode<T> theRoot);
	}
	
	/**
	 * Iteratres through the values of the tree IN order
	 * @author Taylor G Smith
	 *
	 */
	public class BSTInOrderIterator extends BSTIterator {
		public BSTInOrderIterator(BSTNode<T> theRoot) {
			super(theRoot);
		}

		@Override
		protected ArrayList<T> getOrderedValues(BSTNode<T> theRoot) {
			final ArrayList<T> values = new ArrayList<T>();
			return valuesRecurse(theRoot, values);
		}
		
		final private ArrayList<T> valuesRecurse(final BSTNode<T> theRoot, final ArrayList<T> coll) {
			if(theRoot.hasLeft())
				valuesRecurse(theRoot.left, coll);
			coll.add(theRoot.value);
			if(theRoot.hasRight())
				valuesRecurse(theRoot.right, coll);
			return coll;
		}
	}
	
	/**
	 * Iteratres through the values of the tree in POST order
	 * @author Taylor G Smith
	 *
	 */
	public class BSTPostOrderIterator extends BSTIterator {
		public BSTPostOrderIterator(BSTNode<T> theRoot) {
			super(theRoot);
		}

		@Override
		protected ArrayList<T> getOrderedValues(BSTNode<T> theRoot) {
			final ArrayList<T> values = new ArrayList<T>();
			return valuesRecurse(theRoot, values);
		}
		
		final private ArrayList<T> valuesRecurse(final BSTNode<T> theRoot, final ArrayList<T> coll) {
			if(theRoot.hasLeft())
				valuesRecurse(theRoot.left, coll);
			if(theRoot.hasRight())
				valuesRecurse(theRoot.right, coll);
			coll.add(theRoot.value);
			return coll;
		}
	}
	
	/**
	 * Iteratres through the values of the tree in PRE order
	 * @author Taylor G Smith
	 *
	 */
	public class BSTPreOrderIterator extends BSTIterator {
		public BSTPreOrderIterator(BSTNode<T> theRoot) {
			super(theRoot);
		}

		@Override
		protected ArrayList<T> getOrderedValues(BSTNode<T> theRoot) {
			return (ArrayList<T>) theRoot.values();
		}
	}
}
