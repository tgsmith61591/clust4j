package com.clust4j.utils;

import java.util.Collection;

public interface MutableTree<T> extends Collection<T> {
	public Collection<T> values();
}
