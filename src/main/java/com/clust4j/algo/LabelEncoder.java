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
package com.clust4j.algo;

import java.util.LinkedHashSet;
import java.util.TreeMap;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class LabelEncoder extends BaseModel implements java.io.Serializable {
	private static final long serialVersionUID = 6618077714920820376L;
	
	final int[] rawLabels;
	final int numClasses, n;
	final int[] classes;
	
	private volatile TreeMap<Integer, Integer> encodedMapping = null;
	private volatile TreeMap<Integer, Integer> reverseMapping = null;
	private volatile int[] encodedLabels = null;
	private volatile boolean fit = false;
	
	
	public LabelEncoder(int[] labels) {
		VecUtils.checkDims(labels);
		
		final LinkedHashSet<Integer> unique = VecUtils.unique(labels);
		numClasses = unique.size();
		if(numClasses < 2 && !allowSingleClass()) {
			throw new IllegalArgumentException("y has "+numClasses+" unique class" 
				+ (numClasses!=1?"es":"") + " and requires at least two");
		}
		
		this.rawLabels = VecUtils.copy(labels);
		this.n = rawLabels.length;
		
		int idx = 0;
		this.classes = new int[numClasses];
		for(Integer u: unique) classes[idx++] = u.intValue();
		
		// Initialize mappings
		encodedMapping = new TreeMap<>();
		reverseMapping = new TreeMap<>();
		encodedLabels = new int[n];
	}
	
	
	/**
	 * For subclasses that need to have built-in mappings, 
	 * this hook should be called in the constructor
	 * @param key
	 * @param val
	 */
	protected void addMapping(Integer key, Integer value) {
		encodedMapping.put(key, value);
		reverseMapping.put(value, key);
	}
	
	/**
	 * Whether or not to allow only a single class mapping
	 * @return true if allow single class mappings
	 */
	protected boolean allowSingleClass() {
		return false;
	}
	
	@Override
	public LabelEncoder fit() {
		synchronized(fitLock) {
			if(fit) 
				return this;
			
			int nextLabel = 0, label;
			Integer val;
			for(int i = 0; i < n; i++) {
				label = rawLabels[i];
				val = encodedMapping.get(label);
				
				if(null == val) { // not yet seen
					val = nextLabel++;
					encodedMapping.put(label, val);
					reverseMapping.put(val, label);
				}
				
				encodedLabels[i] = val;
			}
			
			
			fit = true;
			return this;
		}
	}
	
	public Integer encodeOrNull(int label) {
		if(!fit) throw new ModelNotFitException("model not yet fit"); 
		return encodedMapping.get(label);
	}
	
	public int[] getClasses() {
		return VecUtils.copy(classes);
	}
	
	public int[] getEncodedLabels() {
		if(!fit) throw new ModelNotFitException("model not yet fit"); 
		return VecUtils.copy(encodedLabels);
	}
	
	public int getNumClasses() {
		return numClasses;
	}
	
	public int[] getRawLabels() {
		return VecUtils.copy(rawLabels);
	}
	
	public Integer reverseEncodeOrNull(int encodedLabel) {
		if(!fit) throw new ModelNotFitException("model not yet fit"); 
		return reverseMapping.get(encodedLabel);
	}
	
	/**
	 * Return an encoded label array back to its original state
	 * @throws IllegalArgumentException if value not in mappings
	 * @return
	 */
	public int[] reverseTransform(int[] encodedLabels) {
		if(!fit) throw new ModelNotFitException("model not yet fit"); 
		final int[] out= new int[encodedLabels.length];
		
		int val; 
		Integer encoding;
		for(int i = 0; i < out.length; i++) {
			val = encodedLabels[i];
			encoding = reverseMapping.get(val);
			
			if(null == encoding)
				throw new IllegalArgumentException(encoding+" does not exist in label mappings");
			out[i] = encoding;
		}
		
		return out;
	}
	
	/**
	 * Encode a new label array based on the fitted mappings
	 * @throws IllegalArgumentException if value not in mappings
	 * @param newLabels
	 * @return
	 */
	public int[] transform(int[] newLabels) {
		if(!fit) throw new ModelNotFitException("model not yet fit"); 
		final int[] out= new int[newLabels.length];
		
		int val; 
		Integer encoding;
		for(int i = 0; i < out.length; i++) {
			val = newLabels[i];
			encoding = encodedMapping.get(val);
			
			if(null == encoding)
				throw new IllegalArgumentException(encoding+" does not exist in label mappings");
			out[i] = encoding;
		}
		
		return out;
	}
}
