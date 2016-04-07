/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
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
	
	public abstract BaseTreeNode<T> getRoot();
	
	/**
	 * Base abstractnode class
	 * @author Taylor G Smith
	 *
	 * @param <T>
	 */
	protected abstract static class BaseTreeNode<T> implements java.io.Serializable {
		private static final long serialVersionUID = 3359038232201737728L;
		
		protected BaseTreeNode() { }
		
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
	
	abstract public Collection<T> values();
}
