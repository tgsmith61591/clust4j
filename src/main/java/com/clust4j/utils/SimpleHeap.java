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

import java.util.ArrayList;

/**
 * Builds a Heap from an ArrayList. Adapted from Python's
 * <a href="https://github.com/python-git/python/blob/master/Lib/heapq.py">heapq</a>
 * priority queue.
 * @author Taylor G Smith
 */
public class SimpleHeap<T extends Comparable<? super T>> extends ArrayList<T> {
	private static final long serialVersionUID = -5346944098593268409L;
	
	public SimpleHeap(ArrayList<T> h) {
		super(h);
		heapifyInPlace(this);
	}
	
	public SimpleHeap(T root) {
		this();
		this.add(root);
	}
	
	public SimpleHeap() {
		// default constructor
		super();
	}
	
	private static <T extends Comparable<? super T>> void heapifyInPlace(final SimpleHeap<T> x) {
		final int n = x.size();
		final int n_2_floor = n / 2;
		
		for(int i = n_2_floor - 1; i >= 0; i--)
			staticSiftUp(x, i);
	}
	
	private static <T extends Comparable<? super T>> void staticSiftDown(final SimpleHeap<T> heap, final int startPos, int pos) {
		T newitem = heap.get(pos);
		
		while(pos > startPos) {
			int parentPos = (pos - 1) >> 1;
			T parent = heap.get(parentPos);
			
			if(newitem.compareTo(parent) < 0) {
				heap.set(pos, parent);
				pos = parentPos;
				continue;
			}
			
			break;
		}
		
		heap.set(pos, newitem);
	}
	
	private static <T extends Comparable<? super T>> void staticSiftUp(final SimpleHeap<T> heap, int pos) {
		int endPos = heap.size();
		int startPos= pos;
		T newItem = heap.get(pos);
		
		int childPos = 2*pos + 1;
		while(childPos < endPos) {
			int rightPos = childPos + 1;
			if(rightPos < endPos && !(heap.get(childPos).compareTo(heap.get(rightPos)) < 0))
				childPos = rightPos;
			
			heap.set(pos, heap.get(childPos));
			pos = childPos;
			childPos = 2*pos + 1;
		}
		
		heap.set(pos, newItem);
		staticSiftDown(heap, startPos, pos);
	}
	
	public T pop() {
		final T lastElement = popInPlace(), returnItem;
		
		if(size() > 0) {
			returnItem = get(0);
			set(0, lastElement);
			siftUp(0);
		} else {
			returnItem = lastElement;
		}
		
		return returnItem;
	}
	
	public void push(T item) {
		add(item);
		siftDown(0, size()-1);
	}
	
	public T pushPop(T item) {
		if(get(0).compareTo(item) < 0) {
			T tmp = get(0);
			set(0, item);
			item = tmp;
		}
		
		return item;
	}
	
	public T popInPlace() {
		if(size() == 0)
			throw new IllegalStateException("heap size 0");
		
		final T last = get(size()-1);
		remove(size()-1);
		return last;
	}
	
	public void siftDown(final int startPos, int pos) {
		staticSiftDown(this, startPos, pos);
	}
	
	public void siftUp(int pos) {
		staticSiftUp(this, pos);
	}
}
