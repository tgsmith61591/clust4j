package com.clust4j.utils;

import java.util.Map;

public class EntryPair<K,V> implements Map.Entry<K,V>, java.io.Serializable {
	private static final long serialVersionUID = -8784924835828002971L;
	private final K key;
	private V value;
	
	public EntryPair(final K key, final V value) {
		this.key = key;
		this.value = value;
	}

	@Override
	public K getKey() {
		return key;
	}

	@Override
	public V getValue() {
		return value;
	}

	@Override
	public V setValue(V value) {
		V old = this.value;
		this.value = value;
		return old;
	}
	
	@Override
	public String toString() {
		return "<" + key + ", " + value + ">";
	}
}
