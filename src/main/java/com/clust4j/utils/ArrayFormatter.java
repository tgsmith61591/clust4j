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

import java.util.Arrays;

public abstract class ArrayFormatter {
	public static String arrayToString(byte[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(short[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(int[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(boolean[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(float[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(double[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(long[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(String[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	public static String arrayToString(char[] d) {
		if(null == d)
			return null;
		return arrayToStringFromCA(Arrays.toString(d).toCharArray());
	}
	
	private static String arrayToStringFromCA(final char[] c) {
		final int padding_chars= 6;
		final int max_len = 2 * padding_chars + 3;
		
		if(c.length > max_len) {
			StringBuilder sb = new StringBuilder();
			
			int ws = 0, next_pos = 0;
			while(padding_chars > ws++) {
				sb.append(c[next_pos++]);
			}
			
			sb.append("..."); // le ellipsis
			
			ws = 0;
			next_pos = c.length - padding_chars;
			while(padding_chars > ws++) {
				sb.append(c[next_pos++]);
			}
			
			return sb.toString();
		}
		
		return new String(c);
	}
}
