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

/**
 * An extension of the {@link SafeLabelEncoder} that accounts for the noisey
 * labels that {@link NoiseyClusterer} algorithms can produce.
 * 
 * Since noisey clusterers have a propensity to create
 * predictions of only -1, we need to allow single mappings
 * in this subclass. This avoids the IllegalArgumentException
 * in the super class constructor.
 * @author Taylor G Smith
 */
public class NoiseyLabelEncoder extends SafeLabelEncoder {
	private static final long serialVersionUID = -5898357662470826812L;
	public static final int NOISE_CLASS = NoiseyClusterer.NOISE_CLASS;

	public NoiseyLabelEncoder(int[] labels) {
		super(labels);
		addMapping(NOISE_CLASS, NOISE_CLASS);
	}
	
	@Override
	public NoiseyLabelEncoder fit() {
		return (NoiseyLabelEncoder) super.fit();
	}
}
