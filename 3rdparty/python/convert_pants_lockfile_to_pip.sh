#!/usr/bin/env bash
#!/bin/bash
set -euo pipefail
#
# This is a rather minimal example Argbash potential
# Example taken from http://argbash.readthedocs.io/en/stable/example.html
# ARG_OPTIONAL_SINGLE([extra-index-url])
# ARG_OPTIONAL_BOOLEAN([requirements])
# ARG_POSITIONAL_SINGLE([lockfile],[lockfile to convert],[])
# ARG_POSITIONAL_SINGLE([platform-tags],[platform tags to use],[])
# ARG_POSITIONAL_SINGLE([output-path],["path to place output"],[])
# ARG_POSITIONAL_SINGLE([output-prefix],["prefix for output"],[])
# ARG_HELP([The general script's help msg])

# ARGBASH_GO()
# needed because of Argbash --> m4_ignore([
### START OF CODE GENERATED BY Argbash v2.9.0 one line above ###
# Argbash is a bash code generator used to get arguments parsing right.
# Argbash is FREE SOFTWARE, see https://argbash.io for more info
# Generated online by https://argbash.io/generate

die() {
	local _ret="${2:-1}"
	test "${_PRINT_HELP:-no}" = yes && print_help >&2
	echo "$1" >&2
	exit "${_ret}"
}

begins_with_short_option() {
	local first_option all_short_options='h'
	first_option="${1:0:1}"
	test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - POSITIONALS
_positionals=()
# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_extra_index_url="no"
_arg_requirements="off"

print_help() {
	printf '%s\n' "The general script's help msg"
	printf 'Usage: %s [--extra-index-url <arg>] [--(no-)requirements] [-h|--help] <lockfile> <platform-tags> <output-path> <output-prefix>\n' "$0"
	printf '\t%s\n' "<lockfile>: lockfile to convert"
	printf '\t%s\n' "<platform-tags>: platform tags to use"
	printf '\t%s\n' "<output-path>: \"path to place output\""
	printf '\t%s\n' "<output-prefix>: \"prefix for output\""
	printf '\t%s\n' "-h, --help: Prints help"
}

parse_commandline() {
	_positionals_count=0
	while test $# -gt 0; do
		_key="$1"
		case "$_key" in
		--extra-index-url)
			test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
			_arg_extra_index_url="$2"
			shift
			;;
		--extra-index-url=*)
			_arg_extra_index_url="${_key##--extra-index-url=}"
			;;
		--no-requirements | --requirements)
			_arg_requirements="on"
			test "${1:0:5}" = "--no-" && _arg_requirements="off"
			;;
		-h | --help)
			print_help
			exit 0
			;;
		-h*)
			print_help
			exit 0
			;;
		*)
			_last_positional="$1"
			_positionals+=("$_last_positional")
			_positionals_count=$((_positionals_count + 1))
			;;
		esac
		shift
	done
}

handle_passed_args_count() {
	local _required_args_string="'lockfile', 'platform-tags', 'output-path' and 'output-prefix'"
	test "${_positionals_count}" -ge 4 || _PRINT_HELP=yes die "FATAL ERROR: Not enough positional arguments - we require exactly 4 (namely: $_required_args_string), but got only ${_positionals_count}." 1
	test "${_positionals_count}" -le 4 || _PRINT_HELP=yes die "FATAL ERROR: There were spurious positional arguments --- we expect exactly 4 (namely: $_required_args_string), but got ${_positionals_count} (the last one was: '${_last_positional}')." 1
}

assign_positional_args() {
	local _positional_name _shift_for=$1
	_positional_names="_arg_lockfile _arg_platform_tags _arg_output_path _arg_output_prefix "

	shift "$_shift_for"
	for _positional_name in ${_positional_names}; do
		test $# -gt 0 || break
		eval "$_positional_name=\${1}" || die "Error during argument parsing, possibly an Argbash bug." 1
		shift
	done
}

parse_commandline "$@"
handle_passed_args_count
assign_positional_args 1 "${_positionals[@]}"

# OTHER STUFF GENERATED BY Argbash

### END OF CODE GENERATED BY Argbash (sortof) ### ])
# [ <-- needed because of Argbash

# converts a lockfile generated by pants to a pip-installable requirements.txt file.
if command -v pex3 >/dev/null 2>&1; then
	PEXBIN=pex3
else
	PEXBIN="$HOME/workspace/lumigator/.python/python3.11.9/python/install/bin/pex3"
fi

wheels="WHEELS_DIR|${HOME}/workspace/wheelhouse"

convert() {
	lock=$1
	tags=$2
	outpath=$3
	prefix=$4

	sed '/^\/\//d' "$lock" >requirements.json
	if [[ $_arg_requirements == 'no' ]]; then
		"$PEXBIN" venv create --complete-platform "$tags" --path-mapping "$wheels" --lock requirements.json -d "${outpath}/${prefix}.venv"
	else
		if [[ $_arg_extra_index_url != "no" ]]; then
			echo "adding $_arg_extra_index_url to the output requirements file"
			echo -n "--extra-index-url $_arg_extra_index_url" >"${outpath}/requirements_${prefix}.txt"
			echo "" >>"${outpath}/requirements_${prefix}.txt"
		fi
		"$PEXBIN" lock export --complete-platform "$tags" --path-mapping "$wheels" requirements.json >>"${outpath}/requirements_${prefix}.txt"
	fi
	rm requirements.json
}
# shellcheck disable=SC2154
convert "$_arg_lockfile" "$_arg_platform_tags" "${_arg_output_path}" "${_arg_output_prefix}"
echo "file at ${_arg_output_path}/${_arg_output_prefix}"
# ] <-- needed because of Argbash
