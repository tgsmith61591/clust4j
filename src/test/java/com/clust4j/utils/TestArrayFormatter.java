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

import static org.junit.Assert.*;

import org.junit.Test;

public class TestArrayFormatter {
	static void p(String s) {
		System.out.println(s);
	}

	@Test
	public void testTooSmall() {
		short[]s = new short[]{1,2,3};
		p(ArrayFormatter.arrayToString(s));
		
		byte[] b = new byte[]{1,2,3};
		p(ArrayFormatter.arrayToString(b));
		
		float[]f = new float[]{1,2};
		p(ArrayFormatter.arrayToString(f));
		
		int[]  i = new int[]{1,2,3};
		p(ArrayFormatter.arrayToString(i));
		
		boolean[]be = new boolean[]{true, false};
		p(ArrayFormatter.arrayToString(be));
		
		long[] l = new long[]{1,2,3};
		p(ArrayFormatter.arrayToString(l));
		
		double[] d = new double[]{1,2};
		p(ArrayFormatter.arrayToString(d));
		
		char[] c = new char[]{'a','b'};
		p(ArrayFormatter.arrayToString(c));
		
		String[] st = new String[]{"a","b"};
		p(ArrayFormatter.arrayToString(st));
		p("");
	}
	
	@Test
	public void testBig() {
		short[]s = new short[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(s));
		
		byte[] b = new byte[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(b));
		
		float[]f = new float[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(f));
		
		int[]  i = new int[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(i));
		
		boolean[]be = new boolean[]{true, false, true, false};
		p(ArrayFormatter.arrayToString(be));
		
		long[] l = new long[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(l));
		
		double[] d = new double[]{1,2,3,4,5,6};
		p(ArrayFormatter.arrayToString(d));
		
		char[] c = new char[]{'a','b','c','d','e','f'};
		p(ArrayFormatter.arrayToString(c));
		
		String[] st = new String[]{"a","b","c","d","e","f"};
		p(ArrayFormatter.arrayToString(st));
		p("");
	}

	@Test
	public void testNull() {
		byte[] b = null;
		assertNotNull(ArrayFormatter.arrayToString(new short[]{1,2,3,4,5,6}));
		assertNull(ArrayFormatter.arrayToString(b));
	}
	
	@Test
	public void testAllNull() {
		short[] s = null;
		char[] c = null;
		int[] i = null;
		long[] l = null;
		float[] f = null;
		double[] d = null;
		boolean[] b = null;
		
		assertNull(ArrayFormatter.arrayToString(s));
		assertNull(ArrayFormatter.arrayToString(c));
		assertNull(ArrayFormatter.arrayToString(i));
		assertNull(ArrayFormatter.arrayToString(l));
		assertNull(ArrayFormatter.arrayToString(f));
		assertNull(ArrayFormatter.arrayToString(d));
		assertNull(ArrayFormatter.arrayToString(b));
	}
}
