package com.clust4j.utils;

import java.util.Collection;

public interface MutableTree<T> {
	public void add(T t);
	public void addAll(Collection<T> values);
	public boolean remove(T value);
	public boolean removeAll(Collection<T> remove);
	public Collection<T> values();
}
