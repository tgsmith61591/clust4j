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

public class QuadTup<C_ONE, C_TWO, C_THREE, C_FOUR> implements java.io.Serializable {
	private static final long serialVersionUID = -6231517018580071453L;
	
	protected final C_ONE one;
	protected final C_TWO two;
	protected final C_THREE three;
	protected final C_FOUR four;
	
	public QuadTup(C_ONE one, C_TWO two, C_THREE three, C_FOUR four) {
		this.one = one;
		this.two = two;
		this.three = three;
		this.four = four;
	}
	
	@Override public String toString() {
		return "("+one+", "+two+", "+three+", "+four+")";
	}
	
	public C_ONE getFirst() { return one; }
	public C_TWO getSecond() { return two; }
	public C_THREE getThird() { return three; }
	public C_FOUR getFourth() { return four; }
}
