package com.clust4j.utils;

import java.util.Iterator;

public abstract class AbstractBinaryTree<T> 
		extends AbstractTree<T> 
		implements java.io.Serializable, Iterable<T>
{
	private static final long serialVersionUID = 8579121844382472649L;
	
	
	
	/**
	 * Base constructor
	 */
	public AbstractBinaryTree() {
		super();
	}

	
	@Override
	public boolean prune(final BaseTreeNode<T> node) {
		if(!(node instanceof BaseBinaryTreeNode))
			throw new IllegalArgumentException("illegal node type");
		
		BaseBinaryTreeNode<T> bbt = (BaseBinaryTreeNode<T>) node;
		if(bbt.hasLeft() || bbt.hasRight()) {
			bbt.prune();
			return true;
		}
		
		return false;
	}
	
	
	public abstract Iterator<T> inOrderIterator();
	public abstract Iterator<T> postOrderIterator();
	public abstract Iterator<T> preOrderIterator();
	
	
	/**
	 * Base abstract binary node class
	 * @author Taylor G Smith
	 *
	 * @param <T>
	 */
	protected abstract static class BaseBinaryTreeNode<T> extends AbstractTree.BaseTreeNode<T> {
		private static final long serialVersionUID = -7171416187589067055L;
		
		BaseBinaryTreeNode() {
			super();
		}
		
		@Override
		public int size() {
			int this_size = 1;
			if(hasLeft())
				this_size += leftChild().size();
			if(hasRight())
				this_size += rightChild().size();
			return this_size;
		}
		
		abstract boolean hasLeft();
		abstract boolean hasRight();
		abstract BaseBinaryTreeNode<T> leftChild();
		abstract BaseBinaryTreeNode<T> rightChild();
	}
}
