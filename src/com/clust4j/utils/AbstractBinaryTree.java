package com.clust4j.utils;

import java.util.Collection;

public abstract class AbstractBinaryTree<T> implements java.io.Serializable {
	private static final long serialVersionUID = 8579121844382472649L;
	
	
	
	/**
	 * Base constructor
	 */
	public AbstractBinaryTree() { /* Empty for now */ }

	
	
	
	public BaseBinaryTreeNode<T> locate(T value) {
		return null == root() ? null : root().locate(value);
	}
	
	public boolean prune(final BaseBinaryTreeNode<T> node) {
		if(node.hasLeft() || node.hasRight()) {
			node.prune();
			return true;
		}
		
		return false;
	}

	public int size() {
		return null == root() ? 0 : root().size();
	}
	
	
	
	public abstract void add(T value);
	public abstract void addAll(Collection<T> values);
	public abstract boolean remove(T value);
	public abstract boolean removeAll(Collection<T> remove);
	public abstract BaseBinaryTreeNode<T> root();
	public abstract Collection<T> values();
	
	
	
	/**
	 * Base abstract node class
	 * @author Taylor G Smith
	 *
	 * @param <T>
	 */
	protected abstract static class BaseBinaryTreeNode<T> {
		protected T value = null;
		
		BaseBinaryTreeNode(final T value) {
			this.value = value;
		}
		
		public T getValue() {
			return value;
		}
		
		public int size() {
			int this_size = 1;
			if(hasLeft())
				this_size += leftChild().size();
			if(hasRight())
				this_size += rightChild().size();
			return this_size;
		}
		
		abstract protected void add(T t);
		abstract boolean hasLeft();
		abstract boolean hasRight();
		abstract BaseBinaryTreeNode<T> leftChild();
		abstract BaseBinaryTreeNode<T> locate(T value);
		abstract protected void prune();
		abstract BaseBinaryTreeNode<T> rightChild();
		abstract public Collection<T> values();
	}
}
