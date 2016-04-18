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

public abstract class Series<T> {
	final Inequality in;
	
	public Series(Inequality in) {
		this.in = in;
	}
	
	interface Evaluator { public boolean eval(double a, double b); }
	public static enum Inequality implements Evaluator {
		LESS_THAN 					{ @Override public boolean eval(double a, double b){return a < b; } },
		
		/*
		 * This one requires some more convoluted logic...
		 */
		EQUAL_TO	{ 
			@Override 
			public boolean eval(double a, double b) {
				boolean anan = Double.isNaN(a);
				boolean bnan = Double.isNaN(b);
				
				/*
				 * For equal to, need to check on NaNs... user
				 * might be trying to assert all or some are NaN.
				 * This wouldn't make sense for any variation of
				 * < or >, so only need to do this for == and !=
				 */
				if(anan && bnan) {
					return true;
				} else if(anan ^ bnan) {
					return false;
				} else {
					return a == b;
				}
			}
		},
		
		GREATER_THAN				{ @Override public boolean eval(double a, double b){return a > b; } },
		LESS_THAN_OR_EQUAL_TO		{ @Override public boolean eval(double a, double b){return a <= b;} },	
		GREATER_THAN_OR_EQUAL_TO	{ @Override public boolean eval(double a, double b){return a >= b;} },
		NOT_EQUAL_TO				{ @Override public boolean eval(double a, double b){ return !EQUAL_TO.eval(a, b); }},
		;
	}
	
	final public boolean eval(final double a, final double b) { return eval(a, in, b); }
	final public static boolean eval(final double a, final Inequality in, final double b) { return in.eval(a, b); }
	
	abstract public T get();
	abstract public T getRef();
	abstract public boolean all();
	abstract public boolean any();
}
