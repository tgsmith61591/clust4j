package com.clust4j.utils;

import java.util.Collection;

public abstract class AbstractTree<T> implements java.io.Serializable {
	private static final long serialVersionUID = -943541226063843703L;

	public AbstractTree() { /* Empty for now */ }

	
	public BaseTreeNode<T> locate(T value) {
		return null == getRoot() ? null : getRoot().locate(value);
	}

	public int size() {
		return null == getRoot() ? 0 : getRoot().size();
	}
	
	public abstract void add(T value);
	public abstract void addAll(Collection<T> values);
	public abstract BaseTreeNode<T> getRoot();
	public abstract boolean remove(T value);
	public abstract boolean removeAll(Collection<T> remove);
	public abstract boolean prune(BaseTreeNode<T> node);
	public abstract Collection<T> values();
	
	/**
	 * Base abstractnode class
	 * @author Taylor G Smith
	 *
	 * @param <T>
	 */
	protected abstract static class BaseTreeNode<T> implements java.io.Serializable {
		private static final long serialVersionUID = 3359038232201737728L;
		
		BaseTreeNode() { }
		
		abstract protected void add(T t);
		abstract public T getValue();
		abstract BaseTreeNode<T> locate(T value);
		abstract protected void prune();
		abstract public int size();
		abstract public Collection<T> values();
		
		@Override
		public String toString() {
			String[] clazz_parts = this.getClass().toString().split(" ");
			return clazz_parts[1] + ": " + getValue();
		}
	}
}
