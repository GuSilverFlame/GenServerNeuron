defmodule NeuronTest do
  use ExUnit.Case, async: false

  setup do
    {:ok, preset_weights_pid} =
      GenServer.start_link(
        Neuron,
        %{
          activation_function: fn n -> 1 / (1 + :math.pow(:math.exp(1), -n)) end,
          input_weights: [0.05, 0.05],
          bias_weight: 0,
          learning_rate: 0.25
        },
        name: :preset
      )

    {:ok, %{preset_weights: preset_weights_pid}}
  end

  test "initializing with random weights is really random" do
    {:ok, pid_1} =
      GenServer.start_link(
        Neuron,
        %{
          activation_function: fn n -> n / (1 + abs(n)) end,
          number_of_inputs: 4,
          learning_rate: 0.25
        },
        name: :random1
      )

    {:ok, pid_2} =
      GenServer.start_link(
        Neuron,
        %{
          activation_function: fn n -> n / (1 + abs(n)) end,
          number_of_inputs: 4,
          learning_rate: 0.25
        },
        name: :random2
      )

    assert GenServer.call(pid_1, {:calculate_inputs, [1, 1, 1, 1]}) !=
             GenServer.call(pid_2, {:calculate_inputs, [1, 1, 1, 1]})
  end

  test "calculate_inputs call returns the expected result when calculating from preset weights",
       %{preset_weights: pid} do
    assert GenServer.call(pid, {:calculate_inputs, [1, 1]}) == 0.52497918747894
  end

  test "call to :update_weights_from_expected_output will not change weights if the result is the same as the expected",
       %{preset_weights: pid} do
    # call once to populate last_inputs
    previous_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})
    # make sure delta is zero
    assert GenServer.call(pid, {:update_weights_from_expected_output, previous_result}) ==
             {0.0, [0.05, 0.05]}

    # calculate again to see if something caused a change, it shouldn't
    assert GenServer.call(pid, {:calculate_inputs, [1, 1]}) == previous_result
  end

  test "call to :update_weights_from_expected_output will not weights if the result is not the same as the expected",
       %{preset_weights: pid} do
    # call once to populate last_inputs
    previous_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})
    # make sure delta is not zero
    assert GenServer.call(pid, {:update_weights_from_expected_output, previous_result + 1}) !=
             {0.0, [0.05, 0.05]}

    # calculate again to make sure it changed
    assert GenServer.call(pid, {:calculate_inputs, [1, 1]}) != previous_result
  end

  test "each repetition of the training cycle brings the error closer to zero, with not too many cycles",
       %{preset_weights: pid} do
    # call once to populate last_inputs
    first_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})
    # learn from expected result
    {first_error, _} =
      GenServer.call(pid, {:update_weights_from_expected_output, first_result + 1})

    # call again to fill up with the new values
    GenServer.call(pid, {:calculate_inputs, [1, 1]})
    # learn again from expected result
    {second_error, _} =
      GenServer.call(pid, {:update_weights_from_expected_output, first_result + 1})

    assert first_error > second_error
    # a third time
    GenServer.call(pid, {:calculate_inputs, [1, 1]})

    {third_error, _} =
      GenServer.call(pid, {:update_weights_from_expected_output, first_result + 1})

    assert second_error > third_error
  end

  test "call to :update_weights_from_error will not change weights if the error is 0", %{
    preset_weights: pid
  } do
    # call once to populate last_inputs
    previous_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})

    # send message asking to recalculate weights but with error == 0, so nothing should change in the end
    GenServer.call(pid, {:update_weights_from_error, 0})
    # calculate again to see if something caused a change, it shouldn't
    assert GenServer.call(pid, {:calculate_inputs, [1, 1]}) == previous_result
  end

  test "call to :update_weights_from_error will change weights if the error is not 0", %{
    preset_weights: pid
  } do
    # call once to populate last_inputs
    previous_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})

    # send message asking to recalculate weights but with error == 0, so nothing should change in the end
    GenServer.call(pid, {:update_weights_from_error, 1})
    # calculate again to see if something caused a change, it shouldn't
    assert GenServer.call(pid, {:calculate_inputs, [1, 1]}) != previous_result
  end

  test "the results change less when there's a lower absolute error value", %{preset_weights: pid} do
    # call once to populate last_inputs
    first_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})

    # send message asking to recalculate weights but with error == 0, so nothing should change in the end
    GenServer.call(pid, {:update_weights_from_error, 1})
    # calculate again with the updated weights
    second_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})
    # update weights again
    GenServer.call(pid, {:update_weights_from_error, 0.1})
    # recalculate with new weights
    third_result = GenServer.call(pid, {:calculate_inputs, [1, 1]})
    assert abs(first_result - second_result) > abs(second_result - third_result)
  end
end
