defmodule Neuron do
  use GenServer

  @moduledoc """
  A neuron for neural networks, for now it only works with raw numeric data
  """

  @impl true
  def init(%{
        activation_function: activation_function,
        learning_rate: learning_rate,
        input_weights: input_weights,
        bias_weight: bias_weight
      })
      when is_function(activation_function) and is_list(input_weights) and is_number(bias_weight) do
    {:ok,
     %{
       activation_function: activation_function,
       learning_rate: learning_rate,
       last_inputs: [],
       last_output: [],
       input_weights: input_weights,
       bias_weight: bias_weight
     }}
  end

  @impl true
  def init(%{
        activation_function: activation_function,
        number_of_inputs: number_of_inputs,
        learning_rate: learning_rate
      })
      when is_function(activation_function) do
    {:ok,
     %{
       activation_function: activation_function,
       learning_rate: learning_rate,
       last_inputs: [],
       last_output: [],
       input_weights: initialize_weights(number_of_inputs),
       bias_weight: :rand.uniform() * 2 - 1
     }}
  end

  @impl true
  def handle_call({:calculate_inputs, inputs}, _from, state = %{input_weights: weights})
      when is_list(inputs) and length(inputs) == length(weights) do
    weighted_sum =
      inputs
      |> Enum.zip(weights)
      |> Enum.reduce(0, fn {input, weight}, acc -> input * weight + acc end)

    result = state.activation_function.(weighted_sum + state.bias_weight * -1)
    {:reply, result, %{state | last_inputs: inputs, last_output: result}}
  end

  @impl true
  def handle_call({:update_weights_from_error, delta}, _from, state) do
    calculate_weights_from_delta(delta, state)
  end

  @impl true
  def handle_call(
        {:update_weights_from_expected_output, expected_output},
        _from,
        state = %{last_output: last_output}
      ) do
    delta = last_output * (1 - last_output) * (expected_output - last_output)
    calculate_weights_from_delta(delta, state)
  end

  defp calculate_weights_from_delta(
         delta,
         state = %{
           last_inputs: last_inputs,
           input_weights: weights,
           bias_weight: bias_weight,
           learning_rate: learning_rate
         }
       )
       when length(last_inputs) == length(weights) do
    new_bias_weight = bias_weight + learning_rate * delta * -1

    new_weights =
      weights
      |> Enum.zip(last_inputs)
      |> Enum.map(fn {weight, input} -> weight + learning_rate * delta * input end)

    {:reply, {delta, new_weights},
     %{state | input_weights: new_weights, bias_weight: new_bias_weight}}
  end

  defp initialize_weights(number_of_inputs) do
    1..number_of_inputs
    |> Enum.map(fn _ -> :rand.uniform() * 2 - 1 end)
  end
end
