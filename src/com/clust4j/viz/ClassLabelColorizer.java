package com.clust4j.viz;

import java.awt.Color;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.TreeMap;

import com.clust4j.utils.VecUtils;

public class ClassLabelColorizer extends TreeMap<Integer, Color> {
	private static final long serialVersionUID = -5067933832383943177L;

	final int m;
	final private int[] labels;
	final private LinkedHashSet<Integer> uniqueLabels;
	final private Random seed;
	
	
	public ClassLabelColorizer(final int[] labels) {
		this(labels, new Random());
	}
	
	public ClassLabelColorizer(final int[] labels, final Random seed) {
		this.labels = VecUtils.copy(labels);
		this.m = labels.length;
		this.seed = seed;
		
		// Build uniqueLabels LinkedHashSet
		uniqueLabels = new LinkedHashSet<>();
		for(int i: labels)
			uniqueLabels.add(i);
		
		build();
	}
	
	
	private void build() {
		Color color;
		
		for(Integer i: uniqueLabels) {
			float r = seed.nextFloat(), g = seed.nextFloat(), b = seed.nextFloat();
			while( this.values().contains(color = new Color(r,g,b)) ) {
				r = seed.nextFloat(); 
				g = seed.nextFloat(); 
				b = seed.nextFloat();
			}
			
			put(i, color);
		}
	}
	
	public Color[] toColorArray() {
		final Color[] colors = new Color[m];
		for(int i=0; i < m; i++)
			colors[i] = get(labels[i]);
		return colors;
	}
}
