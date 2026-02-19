"""Test fixtures for code extraction tests."""

# Realistic reasoning model output from HumanEval with <think> tags
REASONING_MODEL_OUTPUT = """<think>
Okay, I need to complete the Python function has_close_elements. The function takes a list of numbers and a threshold, and returns True if any two numbers are closer to each other than the threshold. Let me think about how to approach this.

First, the problem is to check if there exists at least one pair of elements in the list where the absolute difference between them is less than the given threshold. So, the straightforward way is to compare every possible pair and check their differences.

But wait, if the list is large, checking all pairs could be O(n²) time, which might not be efficient. However, given that the problem doesn't mention performance constraints, maybe the straightforward approach is acceptable here. Let me consider the examples given.

In the first example, [1.0, 2.0, 3.0] with threshold 0.5. The differences between consecutive elements are 1.0 and 1.0, which are both larger than 0.5, so return False. The second example has [1.0, 2.8, 3.0, 4.0, 5.0, 2.0] and threshold 0.3. Let's see: 2.8 and 3.0 have a difference of 0.2, which is less than 0.3, so return True.

So the approach would be to iterate through all possible pairs and check if any of them have a difference less than the threshold. But how to implement this efficiently?

Alternatively, maybe sorting the list first would help. Because if the list is sorted, then for each element, we only need to check with the next elements, since the closest elements would be adjacent after sorting. That would reduce the time complexity to O(n log n) due to sorting, which is better for larger lists.

Let me think: if I sort the list, then for each element, compare it with the next one. If any adjacent pair has a difference less than the threshold, return True. Otherwise, after checking all adjacent pairs, return False.

Wait, but what if there's a pair that's not adjacent but has a smaller difference? For example, if the list is [1, 3, 4], threshold 1.5. The sorted list is [1,3,4]. The differences between 1 and 3 is 2, which is larger than 1.5. Between 3 and 4 is 1, which is less than 1.5. So the adjacent check would catch it. But if the list is [1, 2.5, 3], threshold 0.6. The sorted list is [1, 2.5, 3]. The differences between 1 and 2.5 is 1.5, which is larger than 0.6. Between 2.5 and 3 is 0.5, which is less than 0.6. So again, adjacent pairs would catch it. So sorting the list and checking adjacent pairs is sufficient. Because if there's any pair with a difference less than the threshold, then after sorting, there must be at least one adjacent pair that meets this condition. Wait, is that true?

Let me think of a counterexample. Suppose we have a list [1, 2, 3.5], threshold 0.6. The sorted list is [1, 2, 3.5]. The differences between 1 and 2 is 1, which is larger than 0.6. Between 2 and 3.5 is 1.5, also larger. But if there's a pair like 1 and 3.5, their difference is 2.5, which is larger. So in this case, no pairs. But suppose another example: [1, 1.4, 2.0], threshold 0.5. The sorted list is [1, 1.4, 2.0]. The differences between 1 and 1.4 is 0.4 < 0.5, so it's caught. But if the list was [1, 1.6, 2.0], threshold 0.5. The adjacent pairs are 1-1.6 (0.6) and 1.6-2.0 (0.4). So the second pair would be caught. So in this case, even though the first pair is not within the threshold, the second pair is. So the approach of checking adjacent pairs after sorting would work.

But what if there's a non-adjacent pair that's closer? For example, sorted list [1, 2, 2.5, 3]. Suppose threshold is 0.6. The adjacent pairs are 1-2 (1), 2-2.5 (0.5), 2.5-3 (0.5). So the 2-2.5 and 2.5-3 pairs would be caught. But what if there's a pair like 1 and 2.5? Their difference is 1.5, which is larger than 0.6. So no. So the adjacent pairs are sufficient.

Wait, but suppose we have a list like [1, 3, 4, 5], threshold 2. The sorted list is [1,3,4,5]. The adjacent pairs are 1-3 (2), 3-4 (1), 4-5 (1). So 3-4 and 4-5 are within threshold. But if the threshold was 1.5, then 3-4 is 1, which is less than 1.5. So the approach works.

So the conclusion is that after sorting the list, checking adjacent elements is sufficient. Therefore, the algorithm can be:

Sort the list.

Iterate through each consecutive pair, compute the difference. If any difference is less than the threshold, return True.

If after checking all pairs none meet the condition, return False.

This approach would be O(n log n) time due to sorting, which is better than O(n²) for large lists.

So the steps for the function:

1. Sort the input list.

2. Iterate through the list from index 0 to len(numbers) - 2:

   For each i, compute numbers[i+1] - numbers[i]. If this difference is less than threshold, return True.

3. After checking all pairs, return False.

But wait, since the list is sorted, the difference between numbers[i+1] and numbers[i] is the absolute difference. So no need for absolute value, since the list is sorted in ascending order.

So, the code would be:

Sort the list.

for i in range(len(numbers) - 1):

    if numbers[i+1] - numbers[i] < threshold:

        return True

return False

But what if the list is empty or has only one element? Well, the function is supposed to return False in those cases, since there are no pairs. So the code should handle that.

So, in code:

def has_close_elements(numbers: List[float], threshold: float) -> bool:

    numbers_sorted = sorted(numbers)

    for i in range(len(numbers_sorted) - 1):

        if numbers_sorted[i+1] - numbers_sorted[i] < threshold:

            return True

    return False

Wait, but what if the threshold is zero or negative? The problem says threshold is a float. But according to the examples, the threshold is positive. However, the function's parameters don't have any constraints. So, if threshold is zero or negative, the function would return True if any two elements are equal (for threshold zero) or if any two elements are closer than a negative number (which is impossible, since differences are non-negative). So, perhaps the function should handle that. But according to the problem statement, the threshold is probably intended to be a positive value. However, the code as written would return True if the threshold is negative, because numbers_sorted[i+1] - numbers_sorted[i] is non-negative, and if threshold is negative, then the condition would be true. But that's probably not intended. However, the problem's examples use positive thresholds. So perhaps the function should handle that, but the code as written would handle it. But maybe the function should check if the threshold is non-positive and return False? Or perhaps the problem expects that the threshold is positive. But the problem's docstring doesn't specify. However, the user hasn't provided any constraints, so the code should handle all cases.

But according to the problem's examples, the threshold is positive. So, perhaps the code is correct as written.

Testing the examples:

First example:

numbers = [1.0, 2.0, 3.0], threshold 0.5.

sorted numbers is [1.0, 2.0, 3.0]

differences between consecutive elements are 1.0 and 1.0. Both are >= 0.5. So returns False. Which matches the first example.

Second example:

numbers = [1.0, 2.8, 3.0, 4.0, 5.0, 2.0], threshold 0.3.

sorted numbers is [1.0, 2.0, 2.8, 3.0, 4.0, 5.0]

Check consecutive differences:

2.0 - 1.0 = 1.0 >= 0.3

2.8 - 2.0 = 0.8 >= 0.3

3.0 - 2.8 = 0.2 < 0.3 → returns True. Which matches the second example.

So the code works for the examples.

What about edge cases?

Case where the list has two elements: [1.0, 1.5], threshold 0.6 → difference is 0.5 < 0.6 → returns True.

Another case where threshold is zero: [1.0, 1.0], threshold 0 → returns True.

If threshold is negative, like -0.5, then the code would return True for any list with at least two elements, since the difference is non-negative. But that's probably not intended. However, the problem's docstring doesn't specify, so the code as written is correct.

So the function body would be:

Sort the list, then check consecutive elements.

So the code would be:

numbers_sorted = sorted(numbers)

for i in range(len(numbers_sorted) - 1):

    if numbers_sorted[i+1] - numbers_sorted[i] < threshold:

        return True

return False

But wait, the function is supposed to return True if any two numbers are closer than the threshold. So this code should work.

Now, I need to write this in Python, inside the function.

But the function is supposed to return a bool. So the code would be:

def has_close_elements(numbers: List[float], threshold: float) -> bool:

    numbers_sorted = sorted(numbers)

    for i in range(len(numbers_sorted) - 1):

        if numbers_sorted[i+1] - numbers_sorted[i] < threshold:

            return True

    return False

But wait, what if the list is empty or has only one element? Then len(numbers_sorted) - 1 would be -1, so the loop doesn't run, and returns False. Which is correct.

Yes, that's correct.

So the function body is as written above.

Now, I need to make sure that the code is correctly written, and that the function is properly implemented.

So the final code inside the function would be:

numbers_sorted = sorted(numbers)

for i in range(len(numbers_sorted) - 1):

    if numbers_sorted[i+1] - numbers_sorted[i] < threshold:

        return True

return False

Yes.

Now, I can test this code with the examples.

First example:

has_close_elements([1.0, 2.0, 3.0], 0.5) → sorted is [1,2,3]. The differences are 1 and 1. Both >= 0.5 → returns False. Correct.

Second example:

has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) → sorted is [1.0, 2.0, 2.8, 3.0, 4.0, 5.0]. The differences are 1.0, 0.8, 0.2, 1.0, 1.0. The 0.2 is less than 0.3 → returns True. Correct.

Another test case: [1.0, 1.5], threshold 0.6 → returns True.

Another test case: [1.0, 2.0], threshold 0.5 → returns False.

So the code should work.

Therefore, the function body is as written.
</think>

numbers_sorted = sorted(numbers)
for i in range(len(numbers_sorted) - 1):
    if numbers_sorted[i+1] - numbers_sorted[i] < threshold:
        return True
return False"""

# Expected extracted code from the above output
REASONING_MODEL_EXPECTED_CODE = """numbers_sorted = sorted(numbers)
for i in range(len(numbers_sorted) - 1):
    if numbers_sorted[i+1] - numbers_sorted[i] < threshold:
        return True
return False"""
