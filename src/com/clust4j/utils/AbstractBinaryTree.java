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

public abstract class AbstractBinaryTree<T> 
		extends AbstractTree<T> 
		implements java.io.Serializable
{
	private static final long serialVersionUID = 8579121844382472649L;
	
	
	
	/**
	 * Base constructor
	 */
	public AbstractBinaryTree() {
		super();
	}

	
	public boolean prune(final BaseBinaryTreeNode<T> node) {
		if(node.hasLeft() || node.hasRight()) {
			node.prune();
			return true;
		}
		
		return false;
	}
	
	
	/**
	 * Base abstract binary node class
	 * @author Taylor G Smith
	 *
	 * @param <T>
	 */
	protected abstract static class BaseBinaryTreeNode<T> extends AbstractTree.BaseTreeNode<T> {
		private static final long serialVersionUID = -7171416187589067055L;
		
		protected BaseBinaryTreeNode() {
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
