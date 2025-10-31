{{/*
Expand the name of the chart.
*/}}
{{- define "coeus.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "coeus.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "coeus.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "coeus.labels" -}}
helm.sh/chart: {{ include "coeus.chart" . }}
{{ include "coeus.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "coeus.selectorLabels" -}}
app.kubernetes.io/name: {{ include "coeus.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: {{ include "coeus.name" . }}
component: semantic-api
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "coeus.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "coeus.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified config map name.
*/}}
{{- define "coeus.configMapName" -}}
{{- printf "%s-config" (include "coeus.fullname" .) }}
{{- end }}

{{/*
Create a default fully qualified secret name.
*/}}
{{- define "coeus.secretName" -}}
{{- printf "%s-secret" (include "coeus.fullname" .) }}
{{- end }}

{{/*
Create a default fully qualified PVC name for model cache.
*/}}
{{- define "coeus.modelCachePVCName" -}}
{{- printf "%s-model-cache" (include "coeus.fullname" .) }}
{{- end }}

{{/*
Create a default fully qualified PVC name for data storage.
*/}}
{{- define "coeus.dataStoragePVCName" -}}
{{- printf "%s-data-storage" (include "coeus.fullname" .) }}
{{- end }}





