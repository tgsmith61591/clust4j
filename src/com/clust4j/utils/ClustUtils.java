package com.clust4j.utils;

import java.util.Comparator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

public class ClustUtils {
	final public static <K,V extends Comparable<? super V>> SortedSet<Map.Entry<K,V>> sortEntriesByValue(Map<K,V> map) {
		return sortEntriesByValue(map, false);
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> 
		SortedSet<Map.Entry<K,V>> sortEntriesByValue(Map<K,V> map, final boolean desc) 
	{
		SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(
			new Comparator<Map.Entry<K,V>>() {
				@Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
					int res = e1.getValue().compareTo(e2.getValue());
					return (res != 0 ? res : 1) * (desc ? -1 : 1);
				}
			}
		);
		
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> 
		SortedSet<Map.Entry<K,V>> sortEntriesByValue(
				Map<K,V> map, 
				final Comparator<Map.Entry<K,V>> cmp) 
	{
		SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(cmp);
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}
}
